#!/bin/bash

cp -r tensorboard_raw/ previous_runs/
cp -r study_data/ previous_runs/
cp -r models/ previous_runs/
cp -r model_checkpoints/ previous_runs/
ls figures/
cp -r figures/ previous_runs/
rm -rf tensorboard_raw/hp_autoencoder
rm -rf model_checkpoints/checkpoint_*hdf5
rm -rf models/best_hyper_param_*
rm -rf figures/*

if [ -d "tensorboard_raw/quick_run" ]
then
  rm -rf tensorboard_raw/quick_run;
fi
