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
gdown --id 1iQ98LlWSUUQJOFsT4fLkltwgC6_KrX40
unzip -qq baseline-graspnet.zip
rm -f baseline-graspnet.zip

# DOPE pretrained-weight
echo "Download DOPE pretrained weight"
gdown --id 1tMRa-MsMjFkQSR4yBn0nHBX6ntsy5kLC
unzip -qq DOPE_pretrained_weight.zip
rm -f DOPE_pretrained_weight.zip

cd ..