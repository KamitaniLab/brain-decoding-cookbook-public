"""Pixel-based evaluation of reconstructed images.

- pixel pattern correlation
- pixel pattern identification
- SSIM (structural similarity)
"""


from typing import Dict, List, Optional, Union

from glob import glob
from itertools import product
import os
from pathlib import Path
import re

from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.pipeline.config import init_hydra_cfg
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
import torch
import torchmetrics


# Main #######################################################################

def recon_eval_image(
        recon_image_dir: Union[str, Path],
        true_image_dir: Union[str, Path],
        output_file: Union[str, Path] = "./quality.pkl.gz",
        subjects: List[Optional[str]] = [None],
        rois: List[Optional[str]] = [None],
        recon_image_prefix: Optional[str] = "recon_image-",
        recon_image_ext: str = "tiff",
        true_image_ext: str = "JPEG",
        single_trial: bool = False,
) -> Union[str, Path]:
    # Check decoded or true
    decoded = subjects[0] is not None and rois[0] is not None

    # Display information
    if decoded:
        print("Subjects: {}".format(subjects))
        print("ROIs:     {}".format(rois))
        print("")
    print("Reconstructed image dir: {}".format(recon_image_dir))
    print("True images dir:         {}".format(true_image_dir))
    print("")

    # Loading data ###########################################################

    # Get recon image size
    if decoded:
        recon_image_files = glob(os.path.join(
            recon_image_dir, subjects[0], rois[0], "*." + recon_image_ext))
    else:
        recon_image_files = glob(os.path.join(
            recon_image_dir, "*." + recon_image_ext))
    if len(recon_image_files) == 0:
        raise RuntimeError("Reconstructed images not found:" + os.path.join(
            recon_image_dir, subjects[0], rois[0], "*." + recon_image_ext))
    img = Image.open(recon_image_files[0])
    image_size = img.size

    # Get true images
    true_image_files = sorted(
        glob(os.path.join(true_image_dir, "*." + true_image_ext)))
    if len(true_image_files) == 0:
        raise RuntimeError("True images not found:", glob(
            os.path.join(true_image_dir, "*." + true_image_ext)))
    true_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in true_image_files
    ]
    true_images = np.stack([
        np.array(Image.open(f).convert("RGB").resize(image_size))
        for f in true_image_files
    ])

    # Evaluating reconstruiction performances ################################
    if os.path.exists(output_file):
        print("Loading {}".format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print("Creating an empty dataframe")
        perf_df = pd.DataFrame(columns=[
            "subject", "roi",
            "pixel profile correlation",
            "pixel pattern correlation",
            "pixel rmse",
            "identification accuracy by pixel pattern correlation",
            "identification accuracy by pixel euclidean distance",
            "ssim",
        ])

    update = False
    for subject, roi in product(subjects, rois):
        print("Subject: {} - ROI: {}".format(subject, roi))

        if len(perf_df.query("subject == '{}' and roi == '{}'".format(subject, roi))) > 0:
            print("Already done. Skipped.")
            continue
        update = True

        # Load images
        if decoded:
            recon_image_files = sorted(glob(os.path.join(
                recon_image_dir, subject, roi, "*." + recon_image_ext
            )))
        else:
            recon_image_files = sorted(glob(os.path.join(
                recon_image_dir, "*." + recon_image_ext
            )))
        recon_images = np.stack([
            np.array(Image.open(f).convert("RGB").resize(image_size))
            for f in recon_image_files
        ])
        print("Total number of recon images:", len(recon_images))

        # Obtain image labels
        if single_trial:
            recon_image_labels = []
            for a in recon_image_files:
                recon_image_label = os.path.splitext(os.path.basename(a))[0]
                recon_image_label = recon_image_label.replace(
                    recon_image_prefix, "")  # remove prefix
                recon_image_label = re.sub(
                    "sample(\d+)-", "", recon_image_label)
                recon_image_labels.append(recon_image_label)
        else:
            recon_image_labels = [
                os.path.splitext(os.path.basename(a))[0].replace(
                    recon_image_prefix, "")  # remove prefix
                for a in recon_image_files
            ]

        # Check the matching of image labels between recon and true images
        if not np.array_equal(recon_image_labels, true_image_labels):
            different_labels = True
            y_index = [np.where(np.array(true_image_labels) == x)[
                0][0] for x in recon_image_labels]
            true_images_sorted = true_images[y_index]
        else:
            different_labels = False
            true_images_sorted = true_images

        # Calculate evaluation metrics
        ssim = get_ssim(recon_images, true_images_sorted)
        pixel_procorr = profile_correlation(recon_images, true_images_sorted)
        pixel_patcorr = pattern_correlation(recon_images, true_images_sorted)
        pixel_rmse = get_rmse(recon_images, true_images_sorted)
        pixel_ident_patcorr = pairwise_identification(
            recon_images, true_images,
            metric="correlation", remove_nan=True,
            remove_nan_dist=True, single_trial=different_labels,
            pred_labels=recon_image_labels, true_labels=true_image_labels
        )
        pixel_ident_euclid = pairwise_identification(
            recon_images, true_images,
            metric="euclidean", remove_nan=True,
            remove_nan_dist=True, single_trial=different_labels,
            pred_labels=recon_image_labels, true_labels=true_image_labels
        )

        print("Mean pixel profile correlation:                             {}".format(np.nanmean(pixel_procorr)))
        print("Mean pixel pattern correlation:                             {}".format(np.nanmean(pixel_patcorr)))
        print("Mean pixel rmse:                                            {}".format(np.nanmean(pixel_rmse)))
        print("Mean pixel identification accuracy by pattern correlation:  {}".format(np.nanmean(pixel_ident_patcorr)))
        print("Mean pixel identification accuracy by euclidean distanaces: {}".format(np.nanmean(pixel_ident_euclid)))
        print("Mean SSIM:                                                  {}".format(np.nanmean(ssim)))

        concat_df = pd.DataFrame({
            "subject": subject,
            "roi":     roi,
            "pixel profile correlation": [pixel_procorr.flatten()],
            "pixel pattern correlation": [pixel_patcorr.flatten()],
            "pixel rmse": [pixel_rmse.flatten()],
            "identification accuracy by pixel pattern correlation": [pixel_ident_patcorr.flatten()],
            "identification accuracy by pixel euclidean distance": [pixel_ident_euclid.flatten()],
            "ssim": [ssim.flatten()],
        })
        perf_df = pd.concat([perf_df, concat_df], ignore_index=True)

    print(perf_df)

    # Save the results
    if update:
        perf_df.to_pickle(output_file, compression="gzip")
        print("Saved {}".format(output_file))

    print("All done")

    return output_file


# Functions #######################################################################

def image_preprocess_to_tensor(images):
    images = torch.tensor(images)
    images = images.permute(0, 3, 1, 2) / 255.0
    return images


def get_ssim(recon_images, true_images):
    recon_images = image_preprocess_to_tensor(recon_images)
    true_images = image_preprocess_to_tensor(true_images)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(
        data_range=1.0, reduction="none")
    return ssim(recon_images, true_images).numpy()


def get_rmse(recon_images, true_images):
    p = recon_images.reshape(recon_images.shape[0], -1)
    t = true_images.reshape(true_images.shape[0], -1)
    # sample間rsmeを計算するため，転置しておく
    rmses = mean_squared_error(
        t.T, p.T, multioutput="raw_values", squared=False)
    return rmses


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
        subjects, rois = None, None

    recon_eval_image(
        recon_image_dir=to_absolute_path(cfg.output.path),
        true_image_dir=to_absolute_path(cfg.evaluation.true_image.path),
        output_file=os.path.join(to_absolute_path(cfg.output.path), "quality.pkl.gz"),
        subjects=subjects,
        rois=rois,
        true_image_ext=cfg.evaluation.true_image.ext,
        recon_image_prefix=cfg.output.prefix,
        recon_image_ext=cfg.output.ext,
        single_trial=cfg.evaluation.get("single_trial", False),
    )
