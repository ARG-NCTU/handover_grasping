#!/usr/bin/env bash

cwd=$PWD

if [ ! -d $cwd/model ]; then
    mkdir model
else
    echo "folder exist"
fi

cd $cwd/model

# Andy Zeng ConvNet
echo "Download Andy Zeng ConvNet pretrained weight"
echo "111111" | sudo -S gdown --id 1L1RUg2kz9nW5XwPdmHoxj7e3p9pDHVrH
unzip -qq baseline-graspnet.zip
rm -f baseline-graspnet.zip

# DOPE pretrained-weight
echo "Download DOPE pretrained weight"
echo "111111" | sudo -S gdown --id 17bqzPhC47LpZhI4qD-fMgyzXm_tPv64X
unzip -qq pretrained_weight.zip
rm -f pretrained_weight.zip

cd ..