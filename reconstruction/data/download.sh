#!/bin/sh

set -Ceu

[ -e decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip ] || wget https://figshare.com/ndownloader/files/40106476 -O decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip
unzip -n decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip

[ -e features-ImageNetTraining-VGG_ILSVRC_19_layers-mean.mat.zip ] || wget https://figshare.com/ndownloader/files/41520621 -O features-ImageNetTraining-VGG_ILSVRC_19_layers-mean.mat.zip
unzip -n features-ImageNetTraining-VGG_ILSVRC_19_layers-mean.mat.zip
