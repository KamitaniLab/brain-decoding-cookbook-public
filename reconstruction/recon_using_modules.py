from __future__ import annotations

from typing import Dict, List, Optional, Union

from glob import glob
from itertools import product
from pathlib import Path
import os
from functools import partial
from torchvision import transforms
from bdpy.dataform import Features, DecodedFeatures
from bdpy.dl.torch.models import layer_map, model_factory
from bdpy.feature import normalize_feature
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.recon.torch.icnn import reconstruct
from bdpy.recon.utils import normalize_image, clip_extreme
import hdf5storage
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig
import PIL.Image
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from bdpy.dl.torch.models import VGG19, layer_map
from bdpy.recon.torch.task import inversion as inversion_module
from bdpy.dl.torch.domain import Domain, IrreversibleDomain
from bdpy.recon.torch.modules import encoder as encoder_module
from bdpy.recon.torch.modules import generator as generator_module
from bdpy.recon.torch.modules import latent as latent_module
from bdpy.recon.torch.modules import critic as critic_module
from bdpy.dl.torch.domain import image_domain
from PIL import Image
from IPython import embed

class ResizeDomain(IrreversibleDomain):
    def __init__(self, image_size, device):
        super().__init__()
        self.image_size = image_size
        self.device = device
    def send(self, images):
        gen_image_size = (images.shape[2], images.shape[3])
        top_left = ((gen_image_size[0] - self.image_size[0]) // 2,
                    (gen_image_size[1] - self.image_size[1]) // 2)

        image_mask = np.zeros(images.shape)
        image_mask[:, :,
                    top_left[0]:top_left[0] + self.image_size[0],
                    top_left[1]:top_left[1] + self.image_size[1]] = 1
        image_mask_t = torch.FloatTensor(image_mask).to(self.device)

        images = torch.masked_select(images, image_mask_t.bool()).view(
            (1, self.image_size[2], self.image_size[0], self.image_size[1])
        )
        return images

def put_features_on(
    features: dict[str, torch.Tensor], *args, **kwargs
) -> dict[str, torch.Tensor]:
    return {k: v.to(*args, **kwargs) for k, v in features.items()}

def recon_icnn_using_modules(
        features_dir: Union[str, Path],
        output_dir: Union[str, Path],
        encoder_cfg: DictConfig,
        subjects: List[Optional[str]] = [None],
        rois: List[Optional[str]] = [None],
        generator_cfg: Optional[DictConfig] = None,
        features_decoders_dir: Optional[Union[str, Path]] = None,
        n_iter: int = 200,
        feature_scaling: Optional[str] = None,
        output_image_ext: str = "tiff",
        output_image_prefix: str = "recon_image-",
        device: str = "cuda:0"
) -> Union[str, Path]:
    
    encoder_layers = encoder_cfg.layers
    layer_mapping = layer_map(encoder_cfg.name)
    layer_names = [layer_mapping[key] for key in encoder_layers]

    dtype = torch.float32

    feature_network = VGG19()
    feature_network.load_state_dict(torch.load(encoder_cfg.parameters_file))
    feature_network.to(device)
    feature_network.eval()

    encoder = encoder_module.SimpleEncoder(
        feature_network=feature_network,
        layer_names=layer_names,
        domain=image_domain.BdPyVGGDomain(device=device, dtype=dtype),
    )
    generator_domain = ResizeDomain(
        image_size=(224, 224, 3),
        device=device
    )

    generator_network = model_factory(generator_cfg.name)
    generator_network.to(device)
    generator_network.load_state_dict(torch.load(generator_cfg.parameters_file))
    generator_network.eval()

    generator = generator_module.DNNGenerator(
        generator_network=generator_network,
        domain=generator_domain
    )

    critic = critic_module.TargetNormalizedMSE()
    optimizer = optim.AdamW(generator.parameters(), lr=0.001)
    scheduler = None
    latent = latent_module.ArbitraryLatent(
        shape=(4096,),
        init_fn=partial(nn.init.normal_, mean=0, std=1),
    )
    latent.to(device)

    callbacks = inversion_module.CUILoggingCallback()


    pipeline = inversion_module.FeatureInversionTask(
        encoder=encoder,
        generator=generator,
        latent=latent,
        critic=critic,
        optimizer=optimizer,
        callbacks=callbacks,
        num_iterations= n_iter
    )
    print('set up')

    for subject, roi in product(subjects, rois):

        decoded = subject is not None and roi is not None

        print("----------------------------------------")
        if decoded:
            print("Subject: " + subject)
            print("ROI:     " + roi)
        print("")

        if decoded:
            save_dir = os.path.join(output_dir, subject, roi)
        else:
            save_dir = os.path.join(output_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images if images is None
        if decoded:
            matfiles = sorted(glob(os.path.join(features_dir, encoder_layers[0], subject, roi, "*.mat")))
        else:
            matfiles = sorted(glob(os.path.join(features_dir, encoder_layers[0], "*.mat")))
        images = sorted([os.path.splitext(os.path.basename(fl))[0] for fl in matfiles])

        # Load DNN features
        if decoded:
            features = DecodedFeatures(features_dir, squeeze=False)
        else:
            features = Features(features_dir)

        # Images loop
        for i, image_label in enumerate(images[:2]):
            print("Image: " + image_label)

            # Districuted computation control
            #snapshots_dir = os.path.join(
            #    save_dir, "snapshots", "image-%s" % image_label)
            #if os.path.exists(snapshots_dir):
            #    print("Already done or running. Skipped.")
            #    continue
            #else:
            #    os.makedirs(snapshots_dir)

            # Load DNN features
            if decoded:
                feat = {
                    layer_mapping[layer]: torch.tensor(features.get(layer=layer, subject=subject, roi=roi, label=image_label)).to(device)
                    for layer in encoder_layers
                }
                # Load bias
                feat_mean0_train = {}
                for layer in encoder_layers:
                    fn = os.path.join(features_decoders_dir, layer, subject, roi, "model/y_mean.mat")
                    feat_mean0_train[layer] = hdf5storage.loadmat(fn)["y_mean"]
            else:
                feat = {
                    layer: features.get(layer=layer, label=image_label)
                    for layer in encoder_layers
                }

            
            pipeline.reset_states()
            generated_image = pipeline(feat)
            generated_image = torch.reshape(generated_image, (1, 224, 224, 3))

            image = Image.fromarray(
                generated_image[0].detach().cpu().numpy().astype(np.uint8)
            )

            image.save(os.path.join(save_dir , f"{image_label}.jpg"))

if __name__ == "__main__":

    cfg = init_hydra_cfg()

    if "decoded_features" in cfg:
        features_dir = to_absolute_path(cfg.decoded_features.path)
        features_decoders_dir = to_absolute_path(cfg.decoded_features.decoders.path)
        subjects = cfg.decoded_features.subjects
        rois = cfg.decoded_features.rois
    elif "features" in cfg:
        features_dir = to_absolute_path(cfg.features.path)
        features_decoders_dir = None
        subjects, rois = [None], [None]

    recon_icnn_using_modules(
        features_dir=features_dir,
        features_decoders_dir=features_decoders_dir,
        output_dir=to_absolute_path(cfg.output.path),
        subjects=subjects,
        rois=rois,
        encoder_cfg=cfg.encoder,
        generator_cfg=cfg.generator,
        n_iter=cfg.icnn.num_iteration,
        feature_scaling=cfg.icnn.get("feature_scaling", None),
        output_image_ext=cfg.output.ext,
        output_image_prefix=cfg.output.prefix,
        device=cfg.get("device", "cuda:0")
    )
