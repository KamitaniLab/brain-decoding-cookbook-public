#!/bin/sh

set -Ceu

[ -e decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip ] || wget https://figshare.com/ndownloader/files/40106476 -O decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip
unzip -n decoded_features-ImageNetTest-deeprecon_originals-VGG19.zip
