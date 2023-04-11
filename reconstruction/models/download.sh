#!/bin/sh

set -Ceu

wget https://figshare.com/ndownloader/files/38225868 -O VGG_ILSVRC_19_layers.zip
unzip VGG_ILSVRC_19_layers.zip
wget https://figshare.com/ndownloader/files/38225829 -O bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip
unzip bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip
