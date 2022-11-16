'''Make figures of feature decoding results.'''


import argparse
from glob import glob
from itertools import product
import os

from bdpy.evals.metrics import pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import pandas as pd
from PIL import Image
import yaml


# Main #######################################################################

def recon_image_eval(
        recon_image_dir,
        true_image_dir,
        output_file='./quality.pkl.gz',
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
):

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Reconstructed image dir: {}'.format(recon_image_dir))
    print('True images dir:         {}'.format(true_image_dir))
    print('')

    # Loading data ###########################################################

    # Get recon image size
    img = Image.open(glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0])
    recon_image_size = img.size

    # True images
    true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
    true_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in true_image_files
    ]

    true_images = np.vstack([
        np.array(Image.open(f).resize(recon_image_size)).flatten()
        for f in true_image_files
    ])

    # Evaluating reconstruiction performances ################################

    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'subject', 'roi', 'pixel correlation', 'identification accuracy'
        ])

    for subject, roi in product(subjects, rois):
        print('Subject: {} - ROI: {}'.format(subject, roi))

        if len(perf_df.query('subject == "{}" and roi == "{}"'.format(subject, roi))) > 0:
            print('Already done. Skipped.')
            continue

        recon_image_files = sorted(glob(os.path.join(
            recon_image_dir, subject, roi, '*.' + recon_image_ext
        )))
        recon_image_labels = [
            os.path.splitext(os.path.basename(a))[0]
            for a in recon_image_files
        ]

        # matching true and reconstructed images
        # TODO: better way?
        if len(recon_image_files) != len(true_image_files):
            raise RuntimeError('The number of true ({}) and reconstructed ({}) images mismatch'.format(
                len(true_image_files),
                len(recon_image_files)
            ))
        for tf, rf in zip(true_image_labels, recon_image_labels):
            if not tf in rf:
                raise RuntimeError(
                    'Reconstructed image for {} not found'.format(tf)
                )

        # Load reconstructed images
        recon_images = np.vstack([
            np.array(Image.open(f)).flatten()
            for f in recon_image_files
        ])

        # Calculate evaluation metrics
        r_pixelt = pattern_correlation(recon_images, true_images)
        ident = pairwise_identification(recon_images, true_images)

        print('Mean pixel correlation:       {}'.format(np.nanmean(r_pixelt)))
        print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

        perf_df = perf_df.append(
            {
                'subject': subject,
                'roi':     roi,
                'pixel correlation': r_pixelt.flatten(),
                'identification accuracy': ident.flatten(),
            },
            ignore_index=True
        )

    print(perf_df)

    # Save the results
    perf_df.to_pickle(output_file, compression='gzip')
    print('Saved {}'.format(output_file))

    print('All done')

    return output_file


# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    with open(conf['feature decoding'], 'r') as f:
        conf_featdec = yaml.safe_load(f)

    conf.update({
        'feature decoding': conf_featdec
    })

    if 'analysis name' in conf['feature decoding']:
        analysis_name = conf['feature decoding']['analysis name']
    else:
        analysis_name = ''

    recon_image_eval(
        os.path.join(conf['recon output dir'], analysis_name),
        conf['true image dir'],
        output_file=os.path.join(conf['recon output dir'], analysis_name, 'quality.pkl.gz'),
        subjects=conf['recon subjects'],
        rois=conf['recon rois']
    )
