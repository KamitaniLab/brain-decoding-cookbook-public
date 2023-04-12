# Reconstruction with decoded DNN features

## Setup

Run `download.sh` in `data` and `models` directories to download and prepare data (decoded DNN features, pretrained VGG-19, and pretrained image generator).

## Usage

iCNN reconstruction:

Run the following command.

``` shellsession
$ python recon_icnn_image_vgg19_dgn_relu7gen_gd.py config/recon_icnn_vgg19_dgn_relu7gen_gd_1000iter_decoded_deeprecon_originals.yaml
```

This will output reconstructed images in `data/reconstruction/icnn/vgg19_dgn_relu7gen_gd_1000iter_pytorch/decoded/deeprecon_originals`.

## Appendix

- Data files are hosted at <https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384>.
