"""DNN feature extraction from images."""


from typing import List, Tuple, Union

from glob import glob
from pathlib import Path
import os

from bdpy.dataform import save_feature
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import layer_map, model_factory
from bdpy.pipeline.config import init_hydra_cfg
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import DictConfig
import PIL.Image
import torch
from tqdm import tqdm


# Main function ##############################################################

def extract_image_features(
        images_dir: Union[str, Path],
        output_dir: Union[str, Path],
        encoder_cfg: DictConfig,
        image_ext: str = "JPEG",
        batch_size: int = 1,
        device: str = "cuda:0"
) -> Union[str, Path]:
    """Extract DNN features from images.

    The features are saved as `<output_dir>/<layer>/<image label>.mat` with the
    array key `feat`, which is the layout read by `bdpy.dataform.Features` and
    thus by the feature decoding scripts.

    Parameters
    ----------
    images_dir: str or Path
        Directory containing the stimulus images.
    output_dir: str or Path
        Feature output directory path.
    encoder_cfg: DictConfig
        Image encoder configuration.
    image_ext: optional, str
        Extension of the image files.
    batch_size: optional, int
        Number of images forwarded at once.
    device: optional, str
        Device on which the encoder runs.
    """

    # Network settings -------------------------------------------------------
    layers = list(encoder_cfg.layers)
    layer_mapping = layer_map(encoder_cfg.name)

    # Input image size; (width, height) as expected by PIL
    input_image_shape = encoder_cfg.input_image_shape
    image_size = (input_image_shape[1], input_image_shape[0])

    print("Encoder:      " + encoder_cfg.name)
    print("Input size:   %d x %d" % image_size)
    print("Layers:       " + ", ".join(layers))

    # Get images -------------------------------------------------------------
    image_files = sorted(glob(os.path.join(images_dir, "*." + image_ext)))
    if len(image_files) == 0:
        raise RuntimeError(
            "No image found in %s (extension: %s)" % (images_dir, image_ext))

    labels = [Path(image_file).stem for image_file in image_files]

    print("Images:       %d" % len(image_files))
    print("Output:       " + str(output_dir))

    # Districuted computation control
    todo = [
        (image_file, label)
        for image_file, label in zip(image_files, labels)
        if not _is_done(output_dir, layers, label)
    ]
    if len(todo) < len(image_files):
        print("Skipped %d images already done." % (len(image_files) - len(todo)))
    if len(todo) == 0:
        print("Nothing to do.")
        return output_dir

    # Initialize DNN ---------------------------------------------------------
    encoder = model_factory(encoder_cfg.name)
    encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_cfg.parameters_file))
    encoder.eval()

    feature_extractor = FeatureExtractor(
        encoder, layers, layer_mapping, device=device, detach=True)

    # Extract features -------------------------------------------------------
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), batch_size)):
            batch = todo[i:i + batch_size]

            # NOTE: the batch is made C-contiguous. `image_preprocess` returns
            # CHW arrays that keep the HWC memory layout, and a non-contiguous
            # input takes a different convolution path in torch, changing the
            # features by rounding error.
            x = np.ascontiguousarray(np.stack([
                image_preprocess(_load_image(image_file, image_size))
                for image_file, _ in batch
            ]))
            features = feature_extractor.run(torch.from_numpy(x).to(device))

            # Save features
            for j, (_, label) in enumerate(batch):
                for layer in layers:
                    # NOTE: the sample axis is kept so that `Features` can
                    # concatenate the files along it.
                    save_feature(
                        features[layer][j:j + 1], output_dir, layer, label)

    print("All done")

    return output_dir


# Functions ##################################################################

def _load_image(
        image_file: Union[str, Path],
        image_size: Tuple[int, int]
) -> np.ndarray:
    """Load an image resized to `image_size`, as an HWC RGB array."""

    img = PIL.Image.open(image_file)

    # NOTE: every image is converted, not only the non-RGB ones, so that
    # grayscale, palette, and CMYK images are all handled. For grayscale images
    # this is equivalent to stacking the single channel three times.
    img = img.convert("RGB")

    # NOTE: the aspect ratio is not preserved; the whole image is squeezed into
    # the input size of the encoder, as in the original Caffe script.
    img = img.resize(image_size, resample=PIL.Image.BICUBIC)

    return np.asarray(img)


def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    """Convert to Caffe's input image layout.

    Same as `image_preprocess` in the iCNN reconstruction code
    (`reconstruction/recon_icnn_image_gd.py`): swap dimensions and colour
    channels (HWC/RGB --> CHW/BGR), then subtract the channel-wise (BGR) mean
    [104, 117, 123].
    """
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def _is_done(
        output_dir: Union[str, Path], layers: List[str], label: str
) -> bool:
    """Return True if the features of `label` are saved for all `layers`."""

    return all([
        os.path.exists(os.path.join(output_dir, layer, label + ".mat"))
        for layer in layers
    ])


# Entry point ################################################################

if __name__ == "__main__":

    cfg = init_hydra_cfg()

    extract_image_features(
        images_dir=to_absolute_path(cfg.images.path),
        output_dir=to_absolute_path(cfg.features.path),
        encoder_cfg=cfg.encoder,
        image_ext=cfg.images.ext,
        batch_size=cfg.get("batch_size", 1),
        device=cfg.get("device", "cuda:0")
    )
