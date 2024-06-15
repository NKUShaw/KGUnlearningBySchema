#!/bin/bash


conda activate xy_py38

cd /home/bhui/ML/xiaoyang/EMNLP2024/openKE/OpenKE


python retrain_transd_FB15K237.py
python retrain_transe_YAGO.py
python retrain_transh_YAGO.py
python retrain_transd_YAGO.py
python retrain_transd_YAGO-Copy1.py
wait
