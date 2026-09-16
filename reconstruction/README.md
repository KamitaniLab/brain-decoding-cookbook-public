# Reconstruction with decoded DNN features

## Setup

### Downloading data

Run the following in `data` directory.

``` shellsession
$ uv run download.py recon_demo
```

This provides the VGG-19 encoder, the BVLC reference CaffeNet statistics, the relu6/relu7
deep generator networks, and pre-computed decoded features for `sub-01`--`sub-03`.

### Data you need to produce yourself

`recon_demo` does not include everything the configs refer to:

| Referred to as | How to obtain it |
| --- | --- |
| `paths.feature_decoders_*` | Train decoders with the [`feature-decoding`](../feature-decoding) pipeline (`featdec_fastl2lir_train.py`). It writes `<layer>/<subject>/<roi>/model/y_mean.mat`, which reconstruction reads to undo the decoder's mean-centering. |
| `paths.decoded_features_caffenet`, `paths.features_caffenet` | Decode/extract CaffeNet features (`feature-decoding`), or download the per-layer feature archives from the [figshare dataset](https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384). |
| `paths.true_images` | The ImageNet stimulus images, needed only for evaluation. |
| `paths.dgn_parameters.{norm1,norm2,pool5,relu3,relu4}` | Not part of the public distribution; these configs are kept for reference. |

Feature decoders are required for **any** decoded-feature reconstruction, not only for
`feature_std_train_mean_center` scaling.

## Configuration

All file locations live in the `paths` config group, so the analysis configs contain no
absolute paths. To run against a different data store, add a file to `config/paths/` and
select it with `--override paths=<name>`.

## Usage

### iCNN reconstruction

Run the following command.

``` shellsession
$ uv run recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml
```

This will output reconstructed images in `./data/reconstruction/icnn/recon_icnn_image_gd_vgg19_relu7generator_scaling_feature_std_train_mean_center_1000iter/decoded/ImageNetTest_deeprecon_VGG19`.

If you want to change the reconstruction parameters at run time, please use `--override` option.

``` shellsession
# Use Shen scaling

$ uv run recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml  --override icnn.feature_scaling=feature_std_shen_original

# Use raw decoded features ('null' is convert to None in Python script.)

$ uv run recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml  --override icnn.feature_scaling=null
```

### Evaluation

When evaluating the reconstructed images, use the `--analysis` option and specify the name of the reconstruction script. 

``` shellsession
$ uv run recon_eval_image.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml --analysis recon_icnn_image_gd
```


## Issues

We have noticed that the code is not functioning properly with the following versions of PyTorch. Currently, we are working on debugging the issue.

- PyTorch 1.9.1
- PyTorch 1.9.0

## Appendix

- Data files are hosted at <https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384>.
- The environment is pinned in `../pyproject.toml` and `../uv.lock` (Python 3.10--3.11, PyTorch >= 2.1).
- The code has also been run in the following environments.
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6
  - Python 3.10 + PyTorch 1.12.1 + CUDA 11.6
