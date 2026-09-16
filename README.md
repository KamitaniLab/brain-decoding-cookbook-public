# Brain Decoding Cookbook

Codebase for brain decoding analysis.

## Contents

| Directory | Contents |
| --- | --- |
| [`feature-decoding`](https://github.com/KamitaniLab/feature-decoding) | DNN feature decoding (git submodule) |
| [`reconstruction`](reconstruction) | iCNN and feature-to-generator (FG) image reconstruction, and their evaluation |
| [`visualization`](visualization) | Mapping bdata voxels back into a brain volume |
| [`data`](data) | Data download script and its file manifest |

## Setup

```shellsession
$ git clone https://github.com/KamitaniLab/brain-decoding-cookbook-public.git
$ cd brain-decoding-cookbook-public
$ git submodule update --init feature-decoding
```

You can setup environment with [uv](https://docs.astral.sh/uv/):

```shellsession
$ uv sync
```

To run visualization code, install the optional dependencies:
```shellsession
$ uv sync --extra viz
```

> **Note**
> If you use CUDA, specify the PyTorch index according to your CUDA version.
> ```toml
> # example
> torch = [
>     { index = "pytorch-cu124", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
> ]
> ...
> [[tool.uv.index]]
> name = "pytorch-cu124"
> url = "https://download.pytorch.org/whl/cu124"
> explicit = true
> ```
> The candidate custom index URLs are listed [here](https://pytorch.org/get-started/previous-versions/).

## Configuration

Reconstruction is configured with [Hydra](https://hydra.cc/). Every file location lives in a
single `paths` config group, so a config can be pointed at a different data store without
touching the analysis configs:

```
reconstruction/config/
├── paths/default.yaml  # where the downloaded models and datasets live
├── encoder/            # encoder networks; refer to ${paths.*}
├── generator/          # deep generator networks; refer to ${paths.*}
└── recon_*.yaml        # analyses; select `paths: default` in their defaults list
```

To run against a different data store, add another file to `paths/` and select it:

```shellsession
$ uv run recon_icnn_image_gd.py config/recon_icnn_vgg19_relu7generator_gd_1000iter_decoded_ImageNet.yaml \
    --override paths=mystore
```

## For private cookbook

Lab-internal configurations (server paths, subject identifiers), unpublished analysis code,
and analyses whose data is not distributed are kept separately, and are cloned into
`private/` (git-ignored here).

See <https://github.com/KamitaniLab/brain-decoding-cookbook-private>.
