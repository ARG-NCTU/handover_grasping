#!/usr/bin/env bash

cwd=$PWD

if [ ! -d $cwd/data ]; then
    mkdir data
else
    echo "folder exist"
fi

cd $cwd/data

# HANet_datasets
echo "111111" | sudo -S gdown --id 1Hx1UiaN_ezl82dM2C9EbVkM-2bPzqTRM
unzip -qq HANet_datasets.zip
rm -f HANet_datasets.zip

# HANet testing datasets
echo "111111" | sudo -S gdown --id 1yAckmX5aeo32iqwSPcyWx5uLlnqQTwzX
unzip -qq HANet_datasets.zip
rm -f HANet_datasets.zip

# HANet Easy testing datasets
echo "111111" | sudo -S gdown --id 1-eqnDHzbCyyjNqGo8mTQh3rAtHhfRhtm
unzip -qq HANet_easy_datasets.zip
rm -f HANet_easy_datasets.zip

# DOPE testing datasets
echo "111111" | sudo -S gdown --id 1X_x2ZghVudqd6Gdi1qzeBxW0V-VdWprm
unzip -qq DOPE_testing.zip
rm -f DOPE_testing.zip

# multi_view sample data
echo "111111" | sudo -S gdown --id 1VIB6O_x1LrNNntmHGkhjvO_YBfaE-zx-
unzip -qq multi_view.zip
rm -f multi_view.zip

cd ..