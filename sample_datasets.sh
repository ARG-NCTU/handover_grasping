#!/usr/bin/env bash

cwd=$PWD

if [ ! -d $cwd/data ]; then
    mkdir data
else
    echo "folder exist"
fi

cd $cwd/data

# HANet_sample_datasets
echo "111111" | sudo -S gdown --id 1eF4i0yrxpTl9fV_UEzRoqNRNXHBjdWcA
unzip -qq HANet_sample_datasets.zip
rm -f HANet_sample_datasets.zip

# HANet_datasets
echo "111111" | sudo -S gdown --id 1Hx1UiaN_ezl82dM2C9EbVkM-2bPzqTRM
unzip -qq HANet_datasets.zip
rm -f HANet_datasets.zip

# multi_view sample data
echo "111111" | sudo -S gdown --id 1VIB6O_x1LrNNntmHGkhjvO_YBfaE-zx-
unzip -qq multi_view.zip
rm -f multi_view.zip

cd ..