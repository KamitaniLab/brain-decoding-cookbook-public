'''DNN Feature decoding - decoders test (prediction) script'''


from __future__ import print_function

from itertools import product
import os
import shutil
from time import time
import argparse

import bdpy
from bdpy.dataform import load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def featdec_fastl2lir_predict(
        fmri_data_files,
        feature_decoders_dir,
        output_dir='./decoded_features',
        rois_list=None, label_key=None,
        features_list=None, feature_index_file=None,
        excluded_labels=[],
        average_sample=True,
        chunk_axis=1
):
    '''Feature prediction.

    Input:

    - fmri_data_files
    - feature_decoder_dir

    Output:

    - output_dir

    Parameters:

    TBA
    '''

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0] + '-' + conf['__filename__']

    features_list = features_list[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data_files.keys()))
    print('ROIs:            %s' % list(rois_list.keys()))
    print('Decoders:        %s' % feature_decoders_dir)
    print('Layers:          %s' % features_list)
    print('')

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in fmri_data_files.items()}

    # Initialize directories -------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for feat, sbj, roi in product(features_list, fmri_data_files, rois_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        # Distributed computation setup
        # -----------------------------
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_dir_prediction = os.path.join(output_dir, feat, sbj, roi)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        makedir_ifnot(results_dir_prediction)

        distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        x = data_brain[sbj].select(rois_list[roi])       # Brain data
        # TODO: Dirty solution. FIXME!
        try:
            x_labels = data_brain[sbj].get_label(label_key)  # Labels
        except ValueError:
            print(f'{label_key} not found in vmap. Select numerical values of {label_key} as labels.')
            x_labels = list(data_brain[sbj].select(label_key).flatten())

        # Averaging brain data
        if average_sample:
            x_labels_unique = np.unique(x_labels)
            x_labels_unique = [lb for lb in x_labels_unique if lb not in excluded_labels]
            x = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0) for lb in x_labels_unique])
        else:
            # Sample No. + Label
            x_labels_unique = ['sample{:06}-{}'.format(i + 1, lb) for i, lb in enumerate(x_labels)]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Model directory
        # ---------------
        model_dir = os.path.join(feature_decoders_dir, feat, sbj, roi, 'model')

        # Preprocessing
        # -------------
        x_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
        x_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
        y_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
        y_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

        x = (x - x_mean) / x_norm

        # Prediction
        # ----------
        print('Prediction')

        start_time = time()

        model = FastL2LiR()

        test = ModelTest(model, x)
        test.model_format = 'bdmodel'
        test.model_path = model_dir
        test.dtype = np.float32
        test.chunk_axis = chunk_axis

        y_pred = test.run()

        print('Total elapsed time (prediction): %f' % (time() - start_time))

        # Postprocessing
        # --------------
        y_pred = y_pred * y_norm + y_mean

        # Save results
        # ------------
        print('Saving results')

        start_time = time()

        # Predicted features
        for i, label in enumerate(x_labels_unique):
            # Predicted features
            feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

            # Save file name
            save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

            # Save
            save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)

        print('Saved %s' % results_dir_prediction)

        print('Elapsed time (saving results): %f' % (time() - start_time))

        distcomp.unlock(analysis_id)

    print('%s finished.' % analysis_basename)

    return output_dir


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

    if 'feature index file' in conf:
        feature_index_file = os.path.join(
            conf['training feature dir'][0],
            conf['network'],
            conf['feature index file']
        )
    else:
        feature_index_file = None

    if 'exclude test label' in conf:
        excluded_labels = conf['exclude test label']
    else:
        excluded_labels = []

    if 'test single trial' in conf:
        average_sample = not conf['test single trial']
    else:
        average_sample = True

    featdec_fastl2lir_predict(
        conf['test fmri'],
        os.path.join(conf['feature decoder dir'], analysis_name, conf['network']),
        output_dir=os.path.join(conf['decoded feature dir'], analysis_name, 'decoded_features', conf['network']),
        rois_list=conf['rois'],
        label_key=conf['label key'],
        features_list=conf['layers'],
        feature_index_file=feature_index_file,
        excluded_labels=excluded_labels,
        average_sample=average_sample,
        chunk_axis=conf['chunk axis']
    )
