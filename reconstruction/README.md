# Reconstruction with decoded DNN features

## Setup

### Setting up environment

You can make Python environment to run the code with [Anaconda](https://anaconda.org/).

```shellsession
$ conda env create -n <env name> -f env.yaml
$ conda activate <env name>
```

### Downloading data

Run the following in `data` directory.

``` shellsession
$ python download.py recon_demo
```

## Usage

iCNN reconstruction:

Run the following command.

``` shellsession
$ python recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml
```

This will output reconstructed images in `data/reconstruction/icnn/`.

## Issues

We have noticed that the code is not functioning properly with the following versions of PyTorch. Currently, we are working on debugging the issue.

- PyTorch 1.9.1
- PyTorch 1.9.0

## Appendix

- Data files are hosted at <https://figshare.com/articles/dataset/brain-decoding-cookbook/21564384>.
- The code was tested in the following environments.
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6
  - Python 3.10 + PyTorch 1.12.1 + CUDA 11.6
  - Python 3.8 + PyTorch 1.7.1 + CUDA 10.1
  - Docker: [pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime](https://hub.docker.com/layers/pytorch/pytorch/1.7.0-cuda11.0-cudnn8-runtime/images/sha256-9cffbe6c391a0dbfa2a305be24b9707f87595e832b444c2bde52f0ea183192f1)
