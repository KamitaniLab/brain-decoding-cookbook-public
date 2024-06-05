"""Binarize features based on means of reference features."""


import os
from itertools import product

from bdpy.dataform import Features
import numpy as np
import hdf5storage
from tqdm import tqdm 


def binarize_features(src: str, dst: str, ref: str = None, scale: float = 0) -> None:
    '''Binaraize features based on means of reference features.

    Parameters
    ----------
    src : str
      Path to source feature directory.
    dst : str
      Path to output directory.
    ref : str, optional, default=None
      Path to reference feature directory. If None, features are binarized
      based on means of themself.
    scale : float, optional, default=0
      Scaling factor. If zero, features are binalized as 1 or -1.

    Returns
    -------
    None
    '''

    print('----------------------------------------')
    print(f'src:   {src}')
    print(f'dst:   {dst}')
    print(f'ref:   {ref}')
    print(f'scale: {scale}')

    sfeat = Features(src)

    layers = sfeat.layers
    labels = sfeat.labels

    if ref is not None:
        rfeat = Features(ref)
    else:
        rfeat = sfeat

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

    # Binarize features for each layer and label
    for layer, label in tqdm(product(layers, labels)):
        output_dir = os.path.join(dst, layer)
        output_file = os.path.join(output_dir, label + '.mat')

        if os.path.exists(output_file):
            continue

        f = sfeat.get(layer=layer, label=label)

        bin_f = np.zeros(f.shape)

        if scale == 0:
            bin_f[f >= ref_mean[layer]] =  1
            bin_f[f <  ref_mean[layer]] = -1
        else:
            bin_f[f >= 0] = ref_mean[layer][f >= 0] + scale * ref_std[layer][f >= 0]
            bin_f[f <  0] = ref_mean[layer][f <  0] - scale * ref_std[layer][f <  0]

        os.makedirs(output_dir, exist_ok=True)
        hdf5storage.write(bin_f, 'feat', output_file, matlab_compatible=True)
        print(f'Saved {output_file}')

    return None


if __name__ == '__main__':

    settings = [
        {
            'src':   './data/features/ImageNetTest/caffe/VGG19',
            'dst':   './data/features/ImageNetTest/caffe/VGG19_bin_trainmean',
            'ref':   './data/features/ImageNetTraining/caffe/VGG19',
            'scale': 0
        },
        {
            'src':   './data/features/ImageNetTraining/caffe/VGG19',
            'dst':   './data/features/ImageNetTraining/caffe/VGG19_bin_trainmean',
            'ref':   './data/features/ImageNetTraining/caffe/VGG19',
            'scale': 0
        },
    ]

    for s in settings:
        binarize_features(
            s['src'],
            s['dst'],
            ref=s['ref'],
            scale=s['scale']
        )
