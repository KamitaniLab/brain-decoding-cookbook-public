'''DNN Feature decoding - decoders test (prediction) script'''


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time

import bdpy
from bdpy.dataform import load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np


# Main #######################################################################

def featdec_fastl2lir_predict(
        fmri_data,
        decoder_path,
        output_dir='./decoded_features',
        rois=None,
        label_key=None,
        features=None,
        feature_index_file=None,
        excluded_labels=[],
        average_sample=True,
        chunk_axis=1,
        analysis_name="feature_prediction"
):
    '''Feature prediction.

    Input:

    - fmri_data
    - feature_decoder_dir

    Output:

    - output_dir

    Parameters:

    TBA
    '''
    features = features[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data.keys()))
    print('ROIs:            %s' % list(rois.keys()))
    print('Decoders:        %s' % decoder_path)
    print('Layers:          %s' % features)
    print('')

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in fmri_data.items()}

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

    for feat, sbj, roi in product(features, fmri_data, rois):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        # Distributed computation setup
        # -----------------------------
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + feat
        results_dir_prediction = os.path.join(output_dir, feat, sbj, roi)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        makedir_ifnot(results_dir_prediction)

        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        x = data_brain[sbj].select(rois[roi])  # Brain data
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
        model_dir = os.path.join(decoder_path, feat, sbj, roi, 'model')

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

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    test_fmri_data = {
        subject["name"]: subject["paths"]
        for subject in cfg["decoded_feature"]["test_fmri"]["subjects"]
    }
    test_target = cfg["decoded_feature"]["target"]["paths"]
    decoder_path = cfg["decoder"]["path"]
    decoded_feature_dir = cfg["decoded_feature"]["path"]
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["decoded_feature"]["test_fmri"]["rois"]
    }
    label_key = cfg["decoded_feature"]["test_fmri"]["label_key"]
    features = cfg["decoded_feature"]["target"]["layers"]

    feature_index_file = cfg.decoder.target.get("index_file", None)

    average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]
    excluded_labels = cfg.decoded_feature.parameters.get("exclude_labels", [])

    featdec_fastl2lir_predict(
        test_fmri_data,
        decoder_path,
        output_dir=decoded_feature_dir,
        rois=rois,
        label_key=label_key,
        features=features,
        feature_index_file=feature_index_file,
        excluded_labels=excluded_labels,
        average_sample=average_sample,
        chunk_axis=cfg["decoder"]["parameters"]["chunk_axis"],
        analysis_name=analysis_name
    )
