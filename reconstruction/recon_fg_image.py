"""Feature-to-generator (FG) reconstruction of images."""


from typing import Dict, List, Optional, Union

import copy
import glob
from itertools import product
from pathlib import Path
import os

from bdpy.dataform import Features, DecodedFeatures
from bdpy.dl.torch.models import model_factory
from bdpy.feature import normalize_feature
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.recon.utils import normalize_image, clip_extreme
import hdf5storage
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig
import PIL.Image
import scipy.io as sio
import torch


# Main function ##############################################################

def recon_fg_image(
        features_dir: Union[str, Path],
        output_dir: Union[str, Path],
        generator_cfg: DictConfig,
        subjects: List[Optional[str]] = [None],
        rois: List[Optional[str]] = [None],
        features_decoders_dir: Optional[Union[str, Path]] = None,
        feature_scaling: Optional[str] = None,
        relu_normalization: Optional[str] = None,
        forwarding_layers: Optional[Dict[str, str]] = False,
        iterative_normalization: Optional[int] = False,
        output_image_ext: str = "tiff",
        output_image_prefix: str = "recon_image-",
        device: str = "cuda:0"
) -> Union[str, Path]:
    """Feature-to-generator (FG) reconstruction of images.

    Parameters
    ----------
    features_dir: str or Path
        Feature directory path.
    output_dir: str or Path
        Output directory path.
    generator_cfg: DictConfig
        Image generator configuration.
    subjects: list
        Subjects. [None] for true featrures.
    rois: list
        ROIs. [None] for true featrures.
    features_decoders_dir: optional, str or Path
        Feature decoder directory. Required for decoded features.
    feature_scaling: optional, str
        Feature scaling method.
    relu_normalization: optipnal, str
        If not None, decoded feature normalization with ReLU features is applied.
        Set the name of ReLU layer from which the SD for normalization is extracted.
    forwarding_layers: optional, dict of {str, str}
        If not None, averaging decoded features with forwarded features from lower layers.
        Set a dictionary (e.g., `{"fc7": "relu7"}`).
        CAUTION: NOT SUPPORTED YET
    iterative_normalization: optional, int
        If not None, iterative normalization is applied.
    output_image_ext: optional, str
        Extension of output files.
    output_image_prefix: optional, str
        Prefix added in output file name.
    """

    # Network settings -------------------------------------------------------

    # Delta degrees of freedom when calculating SD
    # This should be match to the DDoF used in calculating
    # SD of true DNN features (`feat_std0`)
    std_ddof = 1

    # Axis for channel in the DNN feature array
    channel_axis = 0

    # Initialize CNN ---------------------------------------------------------

    # Average image of ImageNet
    img_mean = np.load(generator_cfg.image_mean_file)
    img_mean = np.mean(img_mean, axis=(1, 2)).astype(np.float32)

    # Generator network
    net_gen = model_factory(generator_cfg.name)
    net_gen.to(device)
    net_gen.load_state_dict(torch.load(generator_cfg.parameters_file))
    net_gen.eval()

    # Feature SD estimated from true CNN features of 10000 images
    feat_std0 = sio.loadmat(generator_cfg.feature_std_file)

    # Reconstrucion ----------------------------------------------------------

    for subject, roi in product(subjects, rois):

        decoded = subject is not None and roi is not None

        print("----------------------------------------")
        if decoded:
            print("Subject: " + subject)
            print("ROI:     " + roi)
        print("")

        if decoded:
            save_dir = os.path.join(output_dir, subject, roi)
            matfiles = sorted(glob.glob(os.path.join(
                features_dir, generator_cfg.input_layer, subject, roi, "*.mat")))
            features = DecodedFeatures(os.path.join(features_dir), squeeze=False)
        else:
            save_dir = output_dir
            matfiles = sorted(glob.glob(os.path.join(features_dir, generator_cfg.input_layer, "*.mat")))
            features = Features(features_dir)

        # Make save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images
        images = sorted([os.path.splitext(os.path.basename(fl))[0]
                        for fl in matfiles])

        # Images loop
        for image_label in images:
            print("Image: " + image_label)

            recon_image_mat_file = os.path.join(
                save_dir, output_image_prefix + image_label + ".mat")
            if os.path.exists(recon_image_mat_file):
                print("Already done. Skipped.")
                continue

            # Load DNN features
            if decoded:
                gen_input_feat = features.get(layer=generator_cfg.input_layer, subject=subject, roi=roi, label=image_label)
            else:
                gen_input_feat = features.get(layer=generator_cfg.input_layer, label=image_label)

            # ----------------------------------------
            # Normalization of decoded features
            # ----------------------------------------
            if decoded:
                # Load training mean feature
                feat_mean0_train = hdf5storage.loadmat(os.path.join(features_decoders_dir, generator_cfg.input_layer, subject, roi, "model/y_mean.mat"))["y_mean"]

                # Nan value should be replaced by train mean
                gen_input_feat[np.isnan(gen_input_feat)] = feat_mean0_train[np.isnan(gen_input_feat)]

                # Normalization
                if feature_scaling is None:
                    pass
                elif feature_scaling == "feature_std":
                    gen_input_feat = normalize_feature(
                        gen_input_feat[0],
                        channel_wise_mean=False, channel_wise_std=False,
                        channel_axis=channel_axis,
                        shift="self", scale=np.nanmean(feat_std0[generator_cfg.input_layer]),
                        std_ddof=std_ddof
                    )[np.newaxis]
                elif feature_scaling == "feature_std_train_mean_center":
                    gen_input_feat = gen_input_feat - feat_mean0_train
                    gen_input_feat = normalize_feature(
                        gen_input_feat[0],
                        channel_wise_mean=False, channel_wise_std=False,
                        channel_axis=channel_axis,
                        shift="self", scale=np.nanmean(feat_std0[generator_cfg.input_layer]),
                        std_ddof=std_ddof
                    )[np.newaxis]
                    gen_input_feat = gen_input_feat + feat_mean0_train
                else:
                    raise ValueError(f"Unsupported feature scaling: {feature_scaling}")

                # Normalization with ReLU features
                if relu_normalization:
                    gen_input_feat = np.maximum(gen_input_feat, 0)
                    gen_input_feat = gen_input_feat - feat_mean0_train
                    gen_input_feat = normalize_feature(
                        gen_input_feat[0],
                        channel_wise_mean=True, channel_wise_std=True,
                        channel_axis=channel_axis,
                        shift="self", scale=feat_std0[relu_normalization],
                        std_ddof=std_ddof
                    )
                    gen_input_feat = gen_input_feat + feat_mean0_train

                # Iterative normalization
                if iterative_normalization:
                    n_norm_iter = iterative_normalization
                    feat = copy.copy(gen_input_feat)
                    orig_mean = np.mean(feat)
                    for _ in range(n_norm_iter):
                        feat = (feat - np.mean(feat)) / np.std(feat, ddof=std_ddof) * \
                            feat_std0[generator_cfg.input_layer] + orig_mean
                        feat[feat < 0] = 0
                    gen_input_feat = feat

            # ----------------
            # Reconstruction
            # ----------------
            recon_img_tensor = net_gen(torch.tensor(gen_input_feat).to(device))
            recon_img = recon_img_tensor.cpu().detach().numpy()
            recon_img = recon_img[0, :, :, :]
            recon_img = img_deprocess(recon_img, img_mean)

            # ------------------
            # Save the results
            # ------------------

            # Save the raw reconstructed image data
            sio.savemat(recon_image_mat_file, {"recon_image": recon_img})

            # To better display the image, clip pixels with extreme values (0.02% of
            # pixels with extreme low values and 0.02% of the pixels with extreme high
            # values). And then normalise the image by mapping the pixel value to be
            # within [0,255].
            recon_image_normalized_file = os.path.join(save_dir, output_image_prefix + image_label + "." + output_image_ext)
            PIL.Image.fromarray(normalize_image(clip_extreme(recon_img, pct=4))).save(recon_image_normalized_file)

    print("All done")

    return output_dir


# Functions ##################################################################

def img_deprocess(img, img_mean=np.float32([104, 117, 123])):
    """Convert from Caffe's input image layout."""
    return np.dstack((img + np.reshape(img_mean, (3, 1, 1)))[::-1])


# Entry point ################################################################

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

    recon_fg_image(
        features_dir=features_dir,
        features_decoders_dir=features_decoders_dir,
        subjects=subjects,
        rois=rois,
        generator_cfg=cfg.generator,
        feature_scaling=cfg.fg.get("feature_scaling", None),
        output_dir=to_absolute_path(cfg.output.path),
        output_image_ext=cfg.output.ext,
        output_image_prefix=cfg.output.prefix,
        device="cuda:0"
    )
