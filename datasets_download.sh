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
echo "111111" | sudo -S gdown --id 176X8k45t62jJAqkhwPX4osBMG456ZwpZ
unzip -qq HANet_datasets.zip
rm -f HANet_datasets.zip

# HANet testing datasets
echo "Download HANet testing datasets"
echo "111111" | sudo -S gdown --id 1yAckmX5aeo32iqwSPcyWx5uLlnqQTwzX
unzip -qq HANet_testing.zip
rm -f HANet_testing.zip

# HANet Easy testing datasets
echo "Download HANet Easy testing datasets"
echo "111111" | sudo -S gdown --id 1-eqnDHzbCyyjNqGo8mTQh3rAtHhfRhtm
unzip -qq HANet_easy_datasets.zip
rm -f HANet_easy_datasets.zip

# DOPE testing datasets
echo "Download  DOPE testing datasets"
echo "111111" | sudo -S gdown --id 1X_x2ZghVudqd6Gdi1qzeBxW0V-VdWprm
unzip -qq Dope_testing.zip
rm -f Dope_testing.zip

# multi_view sample data
echo "Download multi_view sample data"
echo "111111" | sudo -S gdown --id 1VIB6O_x1LrNNntmHGkhjvO_YBfaE-zx-
unzip -qq multi_view.zip
rm -f multi_view.zip

# parallel-jaw-grasping-dataset
echo "Download parallel-jaw-grasping-datase"
echo "111111" | sudo -S gdown --id 1UKx5Bf0Wwg1RAr21UOBvibe03DcHvnwc
unzip -qq parallel-jaw-grasping-dataset.zip
rm -f parallel-jaw-grasping-dataset.zip.
mv data parallel-jaw-grasping-dataset

cd ..