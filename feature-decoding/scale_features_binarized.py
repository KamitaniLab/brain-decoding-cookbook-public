import os
from itertools import product

from bdpy.dataform import Features, DecodedFeatures
import numpy as np
import hdf5storage


def scale_features_binarized(src: str, dst: str, ref: str, scale: float = 1) -> None:
    '''
    Parameters
    ----------
    src : str
      Path to source feature directory.
    dst : str
      Path to output directory.
    ref : str
      Path to reference feature directory.
    scale : float, optional, default=1
      Scaling factor.

    Returns
    -------
    None
    '''

    print('----------------------------------------')
    print(f'src:   {src}')
    print(f'dst:   {dst}')
    print(f'ref:   {ref}')
    print(f'scale: {scale}')

    sfeat = DecodedFeatures(src)

    layers   = sfeat.layers
    subjects = sfeat.subjects
    rois     = sfeat.rois
    labels   = sfeat.labels

    rfeat = Features(ref)

    # Calculating unit mean and SD of reference features
    ref_layers = rfeat.layers
    ref_mean = {}
    ref_std = {}
    for layer in ref_layers:
        f = rfeat.get(layer=layer)

        f_mean = np.mean(f, axis=0, keepdims=True)
        f_std = np.std(f, axis=0, ddof=1, keepdims=True)

        ref_mean.update({layer: f_mean})
        ref_std.update({layer: f_std})

    for layer, subject, roi, label in product(layers, subjects, rois, labels):
        output_dir = os.path.join(dst, layer)
        output_file = os.path.join(output_dir, label + '.mat')

        if os.path.exists(output_file):
            continue

        f = sfeat.get(layer=layer, subject=subject, roi=roi, label=label)

        scale_f = np.zeros(f.shape)

        scale_f[f >= 0] = ref_mean[layer][f >= 0] + scale * ref_std[layer][f >= 0]
        scale_f[f <  0] = ref_mean[layer][f <  0] - scale * ref_std[layer][f <  0]

        os.makedirs(output_dir, exist_ok=True)
        hdf5storage.write(scale_f, 'feat', output_file, matlab_compatible=True)
        print(f'Saved {output_file}')

    return None


if __name__ == '__main__':

    settings = [
        {
            'src':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet',
            'dst':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet_bin_trainmean_scaled_sd0.5',
            'ref':   './data/features/ImageNetTraining/bvlc_alexnet',
            'scale': 0.5
        },
        {
            'src':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet',
            'dst':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet_bin_trainmean_scaled_sd1',
            'ref':   './data/features/ImageNetTraining/bvlc_alexnet',
            'scale': 1.0
        },
        {
            'src':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet',
            'dst':   './data/decoded_features/ImageNetTest/deeprecon_cls_svm_alexnet_allunits/bvlc_alexnet_bin_trainmean_scaled_sd2',
            'ref':   './data/features/ImageNetTraining/bvlc_alexnet',
            'scale': 2.0
        },
    ]

    for s in settings:
        scale_features_binarized(
            s['src'],
            s['dst'],
            ref=s['ref'],
            scale=s['scale']
        )
