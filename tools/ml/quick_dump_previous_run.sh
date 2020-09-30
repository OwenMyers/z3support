#!/bin/bash

cp -r tensorboard_raw/ previous_runs/
cp -r study_data/ previous_runs/
cp -r models/ previous_runs/
cp -r model_checkpoints/ previous_runs/
ls figures/
cp -r figures/ previous_runs/
rm -r tensorboard_raw/hp_autoencoder
rm -r model_checkpoints/checkpoint_*hdf5
rm -r models/best_hyper_param_*
rm figures/*.png
