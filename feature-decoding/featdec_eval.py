'''Feature decoding evaluation.'''


from itertools import product
import os
import re

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.pipeline.config import init_hydra_cfg
import hdf5storage
import numpy as np
import pandas as pd


# Main #######################################################################

def featdec_eval(
        decoded_feature_path,
        true_feature_path,
        output_file='./accuracy.pkl.gz',
        subjects=None,
        rois=None,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        average_sample=True,
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_path

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_path))
    print('')
    print('True features (Test): {}'.format(true_feature_path))
    print('')
    print('Layers: {}'.format(features))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################

    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_path, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_path)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_path)

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'layer', 'subject', 'roi',
            'profile correlation', 'pattern correlation', 'identification accuracy'
        ])

    for layer in features:
        print('Layer: {}'.format(layer))

        true_y = features_test.get(layer=layer)
        true_labels = features_test.labels

        for subject, roi in product(subjects, rois):
            print('Subject: {} - ROI: {}'.format(subject, roi))

            if len(perf_df.query(
                    'layer == "{}" and subject == "{}" and roi == "{}"'.format(
                        layer, subject, roi
                    )
            )) > 0:
                print('Already done. Skipped.')
                continue

            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)
            pred_labels = decoded_features.selected_label

            if not average_sample:
                pred_labels = [re.match('sample\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_dir,
                layer, subject, roi,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            r_prof = profile_correlation(pred_y, true_y_sorted)
            r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)

            if average_sample:
                ident = pairwise_identification(pred_y, true_y_sorted)
            else:
                ident = pairwise_identification(pred_y, true_y, single_trial=True, pred_labels=pred_labels, true_labels=true_labels)

            print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
            print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))
            print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

            perf_df = perf_df.append(
                {
                    'layer':   layer,
                    'subject': subject,
                    'roi':     roi,
                    'profile correlation': r_prof.flatten(),
                    'pattern correlation': r_patt.flatten(),
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

    cfg = init_hydra_cfg()

    decoded_feature_path = cfg["decoded_feature"]["path"]
    target_feature_path = cfg["decoded_feature"]["target"]["paths"][0]  # FIXME
    feature_decoder_path = cfg["decoder"]["path"]

    subjects = [s["name"] for s in cfg["decoded_feature"]["test_fmri"]["subjects"]]
    rois = [r["name"] for r in cfg["decoded_feature"]["test_fmri"]["rois"]]
    features = cfg["decoded_feature"]["target"]["layers"]

    feature_index_file = cfg.decoder.target.get("index_file", None)
    average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]

    featdec_eval(
        decoded_feature_path,
        target_feature_path,
        output_file=os.path.join(decoded_feature_path, 'accuracy.pkl.gz'),
        subjects=subjects,
        rois=rois,
        features=features,
        feature_index_file=feature_index_file,
        feature_decoder_dir=feature_decoder_path,
        average_sample=average_sample
    )
