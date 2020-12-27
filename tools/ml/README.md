# Notes:
* Add `Input` argument `channels_first` when using Z3 data and 
  change the input dimensions.
* You will have to manually add periodic edges for periodic padding before real run
* **requires** running the transform notebook to get the data into a format for the auto encoder

# Link Encodings

It is worth trying some different encodings. I can see some benefits 
and problems with the different approached and will probably just 
need to work through them by trying them. Here are some notes and
thoughts.

* Right now I believe the best thing to do is to physically represent links
  in a "location" We will have zeros "represent" the locations on the
  vertices and the centers of the plaquetts. 
* If you do the above you  still have some options for labeling the links.
  - You can use 1, 2, 3, 4, 5 to represent blank, up (N), down (S), right (E),
    left (W) respectively. In this case horizontal links can only  take values
    1, 4, 5. Vertical links can only take values 1, 2, 3
  - You could also use values 1, 2, 3 for blank, up/righ, down/left
    and the context of the position in the matrix which determines if it
    is a horizontal link or a vertical link would distinguish between
    up and right, and down and left.
* Of the above I think the second one would be nicer but I'm not sure if
  you could confuse the encoder to something important. Going to start with
  the first sub-bullet above and after I get something working try the second.
* Working with second approach as of 2020/08/14
* Transformed data is saved as a numpy file (pickle).

# Directories

* `source` for sphinx documents source
* `src` for source code
* `scripts` the highest level run commands meant to be run from the directory of this README
* `tensorboard_raw` will have a subdirectories specified in the settings file.
  This is so different runs can all keep their tensorboard log files in this
  folder without conflicting
* `study_data` will have subdirectories named with the timestamp of the settings
  file also for keeping runs separate based on settings file. "Raw" data files
  are specified in the settings file. These are loaded using
  `MLToolMixin.import_data` in `base.py` and then saved to their training,
  testing, and data labels in the `study_data` location. The use of
  `import_data` and the same itself happens in the `grid_search`.
* `model_checkpoints`: saves the hdf5 model checkpoints with the settings file
  timestamp to distinguish
* `models`: currently the place to save the best hyperparameters, or final
  models. Also saved in a sub-dir with settings timestamp names.
