'''DNN Feature decoding - decoders training script'''


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time
import warnings

import bdpy
from bdpy.bdata.utils import select_data_multi_bdatas, get_labels_multi_bdatas
from bdpy.dataform import Features, save_array
from bdpy.dataform.utils import get_multi_features
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def featdec_fastl2lir_train(
        fmri_data: Dict[str, List[str]],
        features_paths: List[str],
        output_dir: Optional[str] = './feature_decoders',
        rois: Optional[Dict[str, str]] = None,
        num_voxel: Optional[Dict[str, int]] = None,
        label_key: Optional[str] = None,
        features: Optional[List[str]] = None,
        feature_index_file: Optional[str] = None,
        alpha: int = 100,
        chunk_axis: int = 1,
        analysis_name: str = "feature_decoder_training"
):
    '''Feature decoder training.

    Input:

    - fmri_data
    - features_paths

    Output:

    - output_dir

    Parameters:

    TBA

    Note:

    If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    Note that Y[0] should be sample dimension.
    '''
    if rois is None:
        rois = {}
    if features is None:
        features = []

    features = features[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data.keys()))
    print('ROIs:            %s' % list(rois.keys()))
    print('Target features: %s' % features_paths)
    print('Layers:          %s' % features)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: [bdpy.BData(f) for f in data_files] for sbj, data_files in fmri_data.items()}

    if feature_index_file is not None:
        data_features = [Features(f, feature_index=os.path.join(f, feature_index_file)) for f in features_paths]
    else:
        data_features = [Features(f) for f in features_paths]

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for feat, sbj, roi in product(features, fmri_data, rois):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])

        # Setup
        # -----
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + feat
        results_dir = os.path.join(output_dir, feat, sbj, roi, 'model')
        makedir_ifnot(results_dir)

        # Check whether the analysis has been done or not.
        info_file = os.path.join(results_dir, 'info.yaml')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = yaml.safe_load(f)
            while info is None:
                warnings.warn('Failed to load info from %s. Retrying...'
                              % info_file)
                with open(info_file, 'r') as f:
                    info = yaml.safe_load(f)
            if '_status' in info and 'computation_status' in info['_status']:
                if info['_status']['computation_status'] == 'done':
                    print('%s is already done and skipped' % analysis_id)
                    continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        x = select_data_multi_bdatas(data_brain[sbj], rois[roi])   # Brain data
        x_labels = get_labels_multi_bdatas(data_brain[sbj], label_key)  # Labels

        # Target features and image labels (file names)
        y_labels = np.unique(x_labels)
        y = get_multi_features(data_features, feat, labels=y_labels)  # Target DNN features

        # Use x that has a label included in y
        x = np.vstack([_x for _x, xl in zip(x, x_labels) if xl in y_labels])
        x_labels = [xl for xl in x_labels if xl in y_labels]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Calculate normalization parameters
        # ----------------------------------

        # Normalize X (fMRI data)
        x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

        # Normalize Y (DNN features)
        y_mean = np.mean(y, axis=0)[np.newaxis, :]
        y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

        # Y index to sort Y by X (matching samples)
        # -----------------------------------------
        y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()

        # Save normalization parameters
        # -----------------------------
        print('Saving normalization parameters.')
        norm_param = {
            'x_mean': x_mean, 'y_mean': y_mean,
            'x_norm': x_norm, 'y_norm': y_norm
        }
        save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
        for sv in save_targets:
            save_file = os.path.join(results_dir, sv + '.mat')
            if not os.path.exists(save_file):
                try:
                    save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                    print('Saved %s' % save_file)
                except Exception:
                    warnings.warn('Failed to save %s. Possibly double running.' % save_file)

        # Preparing learning
        # ------------------
        model = FastL2LiR()
        model_param = {'alpha':  alpha,
                       'n_feat': num_voxel[roi],
                       'dtype': np.float32}

        # Distributed computation setup
        # -----------------------------
        makedir_ifnot('./tmp')
        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

        # Model training
        # --------------
        print('Model training')
        start_time = time()

        train = ModelTraining(model, x, y)
        train.id = analysis_id
        train.model_parameters = model_param

        train.X_normalize = {'mean': x_mean, 'std': x_norm}
        train.Y_normalize = {'mean': y_mean, 'std': y_norm}
        train.Y_sort = {'index': y_index}

        train.dtype = np.float32
        train.chunk_axis = chunk_axis
        train.save_format = 'bdmodel'
        train.save_path = results_dir
        train.distcomp = distcomp

        train.run()

        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    training_fmri = {
        subject["name"]: subject["paths"]
        for subject in cfg["decoder"]["training_fmri"]["subjects"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["decoder"]["training_fmri"]["rois"]
    }
    num_voxel = {
        roi["name"]: roi["num"]
        for roi in cfg["decoder"]["training_fmri"]["rois"]
    }
    label_key = cfg["decoder"]["training_fmri"]["label_key"]

    training_target = cfg["decoder"]["target"]["paths"]
    features = cfg["decoder"]["target"]["layers"]
    feature_index_file = cfg.decoder.target.get("index_file", None)

    decoder_dir = cfg["decoder"]["path"]

    featdec_fastl2lir_train(
        training_fmri,
        training_target,
        output_dir=decoder_dir,
        rois=rois,
        num_voxel=num_voxel,
        label_key=label_key,
        features=features,
        feature_index_file=feature_index_file,
        alpha=cfg["decoder"]["parameters"]["alpha"],
        chunk_axis=cfg["decoder"]["parameters"]["chunk_axis"],
        analysis_name=analysis_name
    )
