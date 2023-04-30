# DNN feature decoding

This repository provides scripts of deep neural network (DNN) feature decoding from fMRI brain activities, originally proposed by [Horikawa & Kamitani (2017)](https://www.nature.com/articles/ncomms15037) and employed in DNN-based image reconstruction methods of [Shen et al. (2019)](http://dx.doi.org/10.1371/journal.pcbi.1006633) as well as recent studies in Kamitani lab.

## Usage

### Decoding with PyFastL2LiR

Example config file: [deeprecon_vgg19_allunits.yaml](config/deeprecon_vgg19_allunits.yaml)

- Training: `featdec_fastl2lir_train.py`
- Test (prediction): `featdec_fastl2lir_predict.py`
- Evaluation: `featdec_eval.py`

### Decoding with generic regression models

Example config file: [deeprecon_fmriprep_rep5_allunits_sklearnRidge_alpha100_caffe_alexnet.yaml](config/deeprecon_fmriprep_rep5_allunits_sklearnRidge_alpha100_caffe_alexnet.yaml)

- Training: `featdec_sklearnRidge_train.py` (example for scikit-learn Ridge regression)
- Test (prediction): `featdec_predict.py`
- Evaluation: `featdec_eval.py`

### Decoding with classification

TBA

## References

- Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. *Nature Communications* 8:15037. https://www.nature.com/articles/ncomms15037
- Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. *PLOS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1006633
