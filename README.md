# Handover Affordance Grasping
> Summary description here.


This file will become your README and also the index of your documentation.

## Note for nbdev setup

Jupyter Notebook Environment was setup in Anaconda.
The Anaconda's default nbdev version is 0.2.40
Make sure that you have the latest nbdev install
```
pip3 install -U nbdev
```

Also be careful about the settings.ini.
* name of repo: no dashes. Use underscore, such as handover_grasping.
* user = ARG-NCTU 
* repo_name = handover_grasping

This will generate a setup.py that point to the repo: ARG-NCTU/handover_grasping for docs

You should not modify README.md directly. Do it in index.ipynb

Run nbdev_build_lib to convert *.ipynb to *.py
```
nbdev_build_lib
```

Run nbdev_build_docs to generate documentations in /docs
```
nbdev_build_docs
```

## How to use

Run docker

```
source Docker/docker_run.sh
```

Install

```
source install.sh
```

Download HANet Datasets

```
source sample_datasets.sh
```

Run HANet inference

```
cd example
python3 inference.py
```
