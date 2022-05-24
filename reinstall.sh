#!/usr/bin/env bash
cd /handover_grasping
echo 111111 | sudo -S rm -r /usr/local/lib/python3.6/dist-packages/handover_grasping-0.0.1-py3.6.egg
nbdev_build_lib
echo 111111 | sudo -S python3 setup.py install