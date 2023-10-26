# Visual image reconstruction (Miyawaki et al., 2008)

Python implementation of visual image reconstruction ([Miyawaki et al., 2008](https://doi.org/10.1016/j.neuron.2008.11.004)).

[Codebase for DNN feature decoding](../feature-decoding) was adopted for the modular decoding of local contrast.

## Usage

### Training of local decoders

``` shellsession
$ python featdec_cls_smlr_train.py config/recon_smlr_100voxelselection.yaml
```

### Prediction of local contrast

``` shellsession
$ python featdec_cls_predict.py config/recon_smlr_100voxelselection.yaml
```

### Reconstruction

See [recon_images.ipynb>](./recon_images.ipynb)
