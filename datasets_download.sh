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
echo "111111" | sudo -S gdown --id 1tBPq3BVQ7Pw5aGd2RkcazpSfcmXufsFG
unzip -qq HANet_training_datasets.zip
rm -f HANet_training_datasets.zip

# HANet testing datasets
echo "Download HANet testing datasets"
echo "111111" | sudo -S gdown --id 1YBSPzS4RS7ml5OGjDbKQA8TP3b_OmIlP
unzip -qq HANet_testing.zip
rm -f HANet_testing.zip

# HANet Easy testing datasets
echo "Download HANet Easy testing datasets"
echo "111111" | sudo -S gdown --id 1_ACfjnxUafW3v2oYk3NqSbKWTDPMJT4k
unzip -qq HANet_easy_datasets.zip
rm -f HANet_easy_datasets.zip

# DOPE testing datasets
echo "Download  DOPE testing datasets"
echo "111111" | sudo -S gdown --id 1qA4yZcSXaHxOS_rIWArdiNgWrb8zp8CP
unzip -qq Dope_testing.zip
rm -f Dope_testing.zip

# multi_view sample data
echo "Download multi_view sample data"
echo "111111" | sudo -S gdown --id 1P4w8N1KAgrAMKU1LY0_jFuTPZ7pIX6jn
unzip -qq multi_view.zip
rm -f multi_view.zip

# parallel-jaw-grasping-dataset
echo "Download parallel-jaw-grasping-datase"
echo "111111" | sudo -S gdown --id 18UlspjFeJzrL7aKiBa0Bh2qHW3WdfHrK

unzip -qq parallel-jaw-grasping-dataset.zip
rm -f parallel-jaw-grasping-dataset.zip
mv data parallel-jaw-grasping-dataset

cd ..