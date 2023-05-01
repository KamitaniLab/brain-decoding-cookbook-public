'''DNN Feature decoding - decoders training script'''


from __future__ import print_function

from itertools import product
import os
import shutil
from time import time
import warnings
import argparse

import bdpy
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.ml.model import EnsembleClassifier
from bdpy.util import makedir_ifnot
import numpy as np
from sklearn.svm import SVC
import yaml


# Main #######################################################################

def featdec_cls_linsvm(
        fmri_data_files,
        features_dir,
        output_dir='./feature_decoders',
        rois_list=None, rois_num_voxel=None,
        label_key=None,
        features_list=None, feature_index_file=None,
        chunk_axis=1
):
    '''Feature decoder training.

    Input:

    - fmri_data_files
    - features_dir

    Output:

    - output_dir

    Parameters:

    TBA

    Note:

    If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    Note that Y[0] should be sample dimension.
    '''

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0] + '-' + conf['__filename__']

    features_list = features_list[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data_files.keys()))
    print('ROIs:            %s' % list(rois_list.keys()))
    print('Target features: %s' % features_dir)
    print('Layers:          %s' % features_list)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in fmri_data_files.items()}

    if feature_index_file is not None:
        data_features = Features(os.path.join(features_dir), feature_index=feature_index_file)
    else:
        data_features = Features(os.path.join(features_dir))

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

    for feat, sbj, roi in product(features_list, fmri_data_files, rois_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        # Setup
        # -----
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
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
        x = data_brain[sbj].select(rois_list[roi])       # Brain data
        x_labels = data_brain[sbj].get_label(label_key)  # Labels

        # Target features and image labels (file names)
        y_labels = np.unique(x_labels)
        y = data_features.get(feat, label=y_labels)  # Target DNN features

        # Use x that has a label included in y
        x = np.vstack([_x for _x, xl in zip(x, x_labels) if xl in y_labels])
        x_labels = [xl for xl in x_labels if xl in y_labels]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Y index to sort Y by X (matching samples)
        # -----------------------------------------
        y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()

        # Preparing learning
        # ------------------
        model = EnsembleClassifier(
            model=SVC(kernel='linear'),
            n_estimators=11,
            n_feat=rois_num_voxel[roi],
            normalize_X=True
        )

        # Distributed computation setup
        # -----------------------------
        makedir_ifnot('./tmp')
        distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

        # Model training
        # --------------
        print('Model training')
        start_time = time()

        train = ModelTraining(model, x, y)
        train.id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        train.Y_sort = {'index': y_index}
        train.dtype = np.float32
        train.chunk_axis = chunk_axis
        train.save_format = 'pickle'
        train.save_path = results_dir
        train.distcomp = distcomp

        train.run()

        print('Total elapsed time (model training): %f' % (time() - start_time))

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
        feature_decoders_dir = os.path.join(conf['feature decoder dir'], conf['analysis name'], conf['network'])
    else:
        feature_decoders_dir = os.path.join(conf['feature decoder dir'], conf['network'])

    if 'feature index file' in conf:
        feature_index_file = os.path.join(
            conf['training feature dir'][0],
            conf['network'],
            conf['feature index file']
        )
    else:
        feature_index_file = None

    featdec_cls_linsvm(
        conf['training fmri'],
        os.path.join(
            conf['training feature dir'][0],
            conf['network']
        ),
        output_dir=feature_decoders_dir,
        rois_list=conf['rois'],
        rois_num_voxel=conf['rois voxel num'],
        label_key=conf['label key'],
        features_list=conf['layers'],
        feature_index_file=feature_index_file,
        chunk_axis=conf['chunk axis']
    )
