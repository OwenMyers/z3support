import datetime
import argparse
import os


def main():
    """
    Run this to create a timestamped settings file

    This settings file will have the following sections

    * ``[Settings]``
    * ``[Data]``
    * ``[Plotting]``

    In the ``[Settings]`` section the user will specify the sets of hyper parameters to be swept over as well as any
    singleton hyper parameters and some genereal settings items. Complete example of items::

        [Settings]
        timestamp=2020-08-08T16:21:45.693965
        L=8
        FEATURE_MAP_START=16
        EPOCHS=32
        # quick run of single param or full param sweep. Use True for testing.
        QUICK_RUN=False
        VERBOSE=False
        # result dir in the tensorboard_raw directory
        TENSORBOARD_SUB_DIR=hp_autoencoder
        BATCH_SIZES=8, 16, 32
        N_LAYERS=2, 3, 4
        FEATURE_MAP_STEPS=2, 4, 8, 16
        STRIDE_SIZES=1
        EARLY_STOPPING_PATIENCE=5
        USE_BATCH_NORMALIZATION=True,False
        USE_DROPOUT=False # optional ,True
        OPTIMIZE_STEP_SIZE=1e-4

    Note that the ``timestamp`` is set by the script. Don't change this.

    In the ``[Data]`` sections the user will specify the different sources of data to include in the training.
    There are two different "modes".

    * One: you have saved physics data (on a lattice) as numpy files and you want to train against the contents of
      those files. In that case  do the following and see the "Transforming Data For Autoencoder" section for
      more details on options and what is done with the data.

    Example::

        [Data]
        DATA1=/full/path/to/data1.npy
        DATA2=/full/path/to/data2.npy

    * Two: you have a way of generating the data on the fly or you want a more complete wrapper for a dataset,
      you will want to prefix with "generator-" and only supply data for "DATA1"

    Example::

        [Data]
        DATA1=generator-MnistDataset

    In the first type of data specification, the lables are determined by the file names. Assumption being that
    Each different file represents a dataset from a different type of system. In the second case you wrap all
    that up in the wrapper class and must be able to provide the lables.

    In section ``[Plotting]`` you can set some plotting options like::

        [Plotting]
        N_FEATURE_MAPS=10

    Which will plot 10 randomly selected feature maps in the final model when ``visualize.py`` is run

    Example:
        Run::

            $ python src/new_setting_file.py -d settings

        If you omit the ``-d`` flag  which specifies a directory to place the new file in, then you will get your
        file in the directory you ran the python command from.

    :return: None
    """
    parser = argparse.ArgumentParser(description='Make a blank time stamped settings file.')
    parser.add_argument('-d', type=str, help='File location', default='./')
    args = parser.parse_args()

    now = datetime.datetime.now()

    datetime_str = now.isoformat()

    with open(os.path.join(args.d, datetime_str + '.conf'), 'w') as f:
        f.write('[Settings]\n')
        f.write(f'timestamp={datetime_str}')


if __name__ == '__main__':
    main()
