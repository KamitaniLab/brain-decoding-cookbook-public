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


## For private cookbook

Lab-internal configurations (server paths, subject identifiers), unpublished analysis code,
and analyses whose data is not distributed are kept separately, and are cloned into
`private/` (git-ignored here).

See <https://github.com/KamitaniLab/brain-decoding-cookbook-private>.
