"""iCNN reconstruction; gradient descent, with image generator."""


from typing import Dict, List, Optional, Union

from glob import glob
from itertools import product
from pathlib import Path
import os

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
import torch.optim as optim


# Main function ##############################################################

def recon_icnn_image_gd(
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
    """The iCNN reconstruction with DGN and gradient descent.

    Parameters
    ----------
    features_dir: str or Path
        Feature directory path.
    output_dir: str or Path
        Output directory path.
    encoder_cfg: DictConfig
        Image encoder configuration.
    generator_cfg: optional, DictConfig
        Image generator configuration.
    subjects: list
        Subjects. [None] for true featrures.
    rois: list
        ROIs. [None] for true featrures.
    features_decoders_dir: optional, str or Path
        Feature decoder directory. Required for decoded features.
    feature_scaling: optional, str
        Feature scaling method.
    output_image_ext: optional, str
        Extension of output files.
    output_image_prefix: optional, str
        Prefix added in output file name.
    """

    # Network settings -------------------------------------------------------
    encoder_layers = encoder_cfg.layers
    layer_mapping = layer_map(encoder_cfg.name)

    # Average image of ImageNet
    image_mean = np.load(encoder_cfg.image_mean_file)
    image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])

    # Delta degrees of freedom when calculating SD
    # This should be match to the DDoF used in calculating
    # SD of true DNN features (`feat_std0`)
    std_ddof = 1

    # Axis for channel in the DNN feature array
    channel_axis = 0

    # Reconstruction options -------------------------------------------------

    opts = {
        # Loss function
        "loss_func": torch.nn.MSELoss(reduction="sum"),

        # The total number of iterations for gradient descend
        "n_iter": n_iter,

        # Learning rate
        "lr": (2., 1e-10),

        # Gradient with momentum
        "momentum": (0.9, 0.9),

        # Pixel decay for each iteration
        "decay": (0.01, 0.01),

        # Use image smoothing or not
        "blurring": False,

        # A python dictionary consists of channels to be selected, arranged in
        # pairs of layer name (key) and channel numbers (value); the channel
        # numbers of each layer are the channels to be used in the loss function;
        # use all the channels if some layer not in the dictionary; setting to None
        # for using all channels for all layers;
        "channels": None,

        # A python dictionary consists of masks for the traget CNN features,
        # arranged in pairs of layer name (key) and mask (value); the mask selects
        # units for each layer to be used in the loss function (1: using the uint;
        # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
        # units if some layer not in the dictionary; setting to None for using all
        # units for all layers;
        "masks": None,

        # Display the information on the terminal for every n iterations
        "disp_interval": 1,
    }

    # Initialize DNN ---------------------------------------------------------

    # Initial features
    initial_gen_feat = np.random.normal(0, 1, generator_cfg.latent_shape)

    # Feature SD estimated from true DNN features of 10000 images
    feat_std0 = sio.loadmat(encoder_cfg.feature_std_file)

    # Feature upper/lower bounds
    upper_bound = np.loadtxt(generator_cfg.latent_upper_bound_file, delimiter=" ")
    upper_bound = upper_bound.reshape(generator_cfg.latent_upper_bound_shape)

    # Setup results directory ------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set reconstruction options ---------------------------------------------
    opts.update({
        # The initial image for the optimization (setting to None will use random noise as initial image)
        "initial_feature": initial_gen_feat,
        "feature_upper_bound": upper_bound,
        "feature_lower_bound": 0.,
    })

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
        for i, image_label in enumerate(images):
            print("Image: " + image_label)

            # Districuted computation control
            snapshots_dir = os.path.join(
                save_dir, "snapshots", "image-%s" % image_label)
            if os.path.exists(snapshots_dir):
                print("Already done or running. Skipped.")
                continue
            else:
                os.makedirs(snapshots_dir)

            # Encoder model
            encoder = model_factory(encoder_cfg.name)
            encoder.to(device)
            encoder.load_state_dict(torch.load(encoder_cfg.parameters_file))
            encoder.eval()

            # Generator model
            generator = model_factory(generator_cfg.name)
            generator.to(device)
            generator.load_state_dict(torch.load(generator_cfg.parameters_file))
            generator.eval()

            # Load DNN features
            if decoded:
                feat = {
                    layer: features.get(layer=layer, subject=subject, roi=roi, label=image_label)
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

            # ----------------------------------------
            # Normalization of decoded features
            # ----------------------------------------
            if decoded:
                for layer, ft in feat.items():
                    if feature_scaling is None:
                        pass
                    elif feature_scaling == "feature_std":
                        ft = normalize_feature(
                            ft[0],
                            channel_wise_mean=False, channel_wise_std=False,
                            channel_axis=channel_axis,
                            shift='self', scale=np.mean(feat_std0[layer]),
                            std_ddof=std_ddof
                        )[np.newaxis]
                    elif feature_scaling == "feature_std_train_mean_center":
                        ft = ft - feat_mean0_train[layer]
                        ft = normalize_feature(
                            ft[0],
                            channel_wise_mean=False, channel_wise_std=False,
                            channel_axis=channel_axis,
                            shift="self", scale=np.mean(feat_std0[layer]),
                            std_ddof=std_ddof
                        )[np.newaxis]
                        ft = ft + feat_mean0_train[layer]
                    else:
                        raise ValueError(f"Unsupported feature scaling: {feature_scaling}")

                    feat.update({layer: ft})

            # Norm of the DNN features for each layer
            feat_norm = np.array(
                [np.linalg.norm(feat[layer]) for layer in encoder_layers],
                dtype="float32"
            )

            # Weight of each layer in the total loss function
            # Use the inverse of the squared norm of the DNN features as the
            # weight for each layer
            weights = 1. / (feat_norm ** 2)

            # Normalise the weights such that the sum of the weights = 1
            weights = weights / weights.sum()
            layer_weights = dict(zip(encoder_layers, weights))

            opts.update({"layer_weights": layer_weights})

            # Reconstruction
            recon_image, loss_hist = reconstruct(
                feat,
                encoder,
                generator=generator,
                layer_mapping=layer_mapping,
                optimizer=optim.SGD,
                image_size=encoder_cfg.input_image_shape,
                crop_generator_output=True,
                preproc=image_preprocess,
                postproc=image_deprocess,
                output_dir=save_dir,
                save_snapshot=True,
                snapshot_dir=snapshots_dir,
                snapshot_interval=10,
                snapshot_ext="jpg",
                snapshot_postprocess=normalize_image,
                return_loss=True,
                device=device,
                **opts
            )

            # Save the raw reconstructed image
            recon_image_mat_file = os.path.join(save_dir, output_image_prefix + image_label + ".mat")
            sio.savemat(recon_image_mat_file, {"recon_image": recon_image, "loss_history": loss_hist})

            # To better display the image, clip pixels with extreme values (0.02% of
            # pixels with extreme low values and 0.02% of the pixels with extreme high
            # values). And then normalise the image by mapping the pixel value to be
            # within [0,255].
            recon_image_normalized_file = os.path.join(save_dir, output_image_prefix + image_label + "." + output_image_ext)
            PIL.Image.fromarray(normalize_image(clip_extreme(recon_image, pct=4))).save(recon_image_normalized_file)

    print("All done")

    return output_dir


# Functions ##################################################################

def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    """Convert to Caffe's input image layout."""
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    """Donvert from Caffe's input image layout."""
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])


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

    recon_icnn_image_gd(
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
