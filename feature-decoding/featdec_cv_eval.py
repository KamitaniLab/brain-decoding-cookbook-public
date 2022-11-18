'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os
import re

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import pandas as pd
import yaml


# Main #######################################################################

def featdec_cv_eval(
        decoded_feature_dir,
        true_feature_dir,
        output_file_pooled='./accuracy.pkl.gz',
        output_file_fold='./accuracy_fold.pkl.gz',
        subjects=None,
        rois=None,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        single_trial=False
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_dir))
    print('')
    print('True features (Test): {}'.format(true_feature_dir))
    print('')
    print('Layers: {}'.format(features))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################

    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_dir, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_dir)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_dir)

    cv_folds = decoded_features.folds

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        perf_df_fold = pd.read_pickle(output_file_fold)
    else:
        print('Creating an empty dataframe')
        perf_df_fold = pd.DataFrame(columns=[
            'layer', 'subject', 'roi', 'fold',
            'profile correlation', 'pattern correlation', 'identification accuracy'
        ])

    true_labels = features_test.labels

    for layer in features:
        print('Layer: {}'.format(layer))
        true_y = features_test.get_features(layer=layer)

        for subject, roi, fold in product(subjects, rois, cv_folds):
            print('Subject: {} - ROI: {} - Fold: {}'.format(subject, roi, fold))

            if len(perf_df_fold.query(
                    'layer == "{}" and subject == "{}" and roi == "{}" and fold == "{}"'.format(
                        layer, subject, roi, fold
                    )
            )) > 0:
                print('Already done. Skipped.')
                continue

            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi, fold=fold)
            pred_labels = decoded_features.selected_label

            if single_trial:
                pred_labels = [re.match('trial_\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_dir,
                layer, subject, roi, fold,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            r_prof = profile_correlation(pred_y, true_y_sorted)
            r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)

            if single_trial:
                ident = pairwise_identification(pred_y, true_y, single_trial=True, pred_labels=pred_labels, true_labels=true_labels)
            else:
                ident = pairwise_identification(pred_y, true_y_sorted)

            print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
            print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))
            print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

            perf_df_fold = perf_df_fold.append(
                {
                    'layer':   layer,
                    'subject': subject,
                    'roi':     roi,
                    'fold':    fold,
                    'profile correlation': r_prof.flatten(),
                    'pattern correlation': r_patt.flatten(),
                    'identification accuracy': ident.flatten(),
                },
                ignore_index=True
            )

    print(perf_df_fold)

    # Save the results (each fold)
    perf_df_fold.to_pickle(output_file_fold, compression='gzip')
    print('Saved {}'.format(output_file_fold))

    print('All done')

    # Pool accuracy
    perf_df_pooled = pd.DataFrame(columns=[
        'layer', 'subject', 'roi',
        'profile correlation', 'pattern correlation', 'identification accuracy'
    ])

    for layer, subject, roi in product(features, subjects, rois):
        q = 'layer == "{}" and subject == "{}" and roi == "{}"'.format(layer, subject, roi)
        r = perf_df_fold.query(q)

        r_prof_pooled = r['profile correlation'].mean()
        r_patt_pooled = np.hstack(r['pattern correlation'])
        ident_pooled  = np.hstack(r['identification accuracy'])

        perf_df_pooled = perf_df_pooled.append(
                {
                    'layer':   layer,
                    'subject': subject,
                    'roi':     roi,
                    'profile correlation': r_prof_pooled.flatten(),
                    'pattern correlation': r_patt_pooled.flatten(),
                    'identification accuracy': ident_pooled.flatten(),
                },
                ignore_index=True
            )

    print(perf_df_pooled)

    # Save the results (pooled)
    perf_df_pooled.to_pickle(output_file_pooled, compression='gzip')
    print('Saved {}'.format(output_file_pooled))

    print('All done')

    return output_file_pooled, output_file_fold


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

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''

    decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name,
        conf['network']
    )

    if 'feature index file' in conf:
        feature_index_file = os.path.join(conf['training feature dir'][0], conf['network'], conf['feature index file'])
    else:
        feature_index_file = None

    if 'test single trial' in conf:
        single_trial = conf['test single trial']
    else:
        single_trial = False

    featdec_cv_eval(
        decoded_feature_dir,
        os.path.join(conf['feature dir'][0], conf['network']),
        output_file_pooled=os.path.join(decoded_feature_dir, 'accuracy.pkl.gz'),
        output_file_fold=os.path.join(decoded_feature_dir, 'accuracy_fold.pkl.gz'),
        subjects=list(conf['fmri'].keys()),
        rois=list(conf['rois'].keys()),
        features=conf['layers'],
        feature_index_file=feature_index_file,
        feature_decoder_dir=os.path.join(
            conf['feature decoder dir'],
            analysis_name,
            conf['network']
        ),
        single_trial=single_trial
    )
