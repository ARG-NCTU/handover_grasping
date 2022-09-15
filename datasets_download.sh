#!/usr/bin/env bash

cwd=$PWD

if [ ! -d $cwd/data ]; then
    mkdir data
else
    echo "folder exist"
fi

cd $cwd/data

# HANet training datasets
echo "Download HANet training datasets"
gdown --id 1tBPq3BVQ7Pw5aGd2RkcazpSfcmXufsFG
unzip -qq HANet_training_datasets.zip
rm -f HANet_training_datasets.zip

# HA-Rotated
echo "Download HA-Rotated"
gdown --id 1Ja7-ZAVERoC-A0aEKCno9x351EqxG7r7
unzip -qq HA-Rotated.zip
rm -f HA-Rotated.zip

# HA-Upright
echo "Download HA-Upright"
gdown --id 1rJGt6uLbXa1dDKmmYMkLtpuWOk8QgJdp
unzip -qq HA-Upright.zip
rm -f HA-Upright.zip

# YCB-Obj
echo "Download  YCB-Obj"
gdown --id 1Pw-6PUnQqsStOQRZ6PWOP19DeivOpo30
unzip -qq YCB-Obj.zip
rm -f YCB-Obj.zip

# # multi_view sample data
echo "Download multi_view sample data"
gdown --id 1P4w8N1KAgrAMKU1LY0_jFuTPZ7pIX6jn
unzip -qq multi_view.zip
rm -f multi_view.zip

# ARC-Cluttered
echo "Download ARC-Cluttered"
gdown --id 16BPFZIsN5i_42KXV6avQEyenAN9Y_xwV
unzip -qq ARC-Cluttered.zip
rm -f ARC-Cluttered.zip

cd ..