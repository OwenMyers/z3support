from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import os
import pickle


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    """Wrapper function to create a LearningRateScheduler with step decay schedule."""

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return new_lr

    return LearningRateScheduler(schedule)


class CustomCallbacks(Callback):
    def __init__(self, model_as_json, save_location):
        """
        Constructor

        :param model: the model that is being trained.
        :param save_location: the location you would like to save information to
        """
        super().__init__()
        self.working_model_as_json = model_as_json
        self.save_location = save_location

    def on_epoch_end(self, epoch, logs=None):
        """
        We need a custom solution to saving the model structure itself. There seems to be a bug or two in TF/keras
        around the loading of anything beyond the weights of a ``Checkpoint``. Even with ``save_weights_only=False``
        I have not successfully been able to load a full model. Here we will save it as json and load the weights
        using the output from the checkpointer.

        Why don't we save the whole model here? Because eventually this bug will be fixed and we can remove this whole
        chunk of code. The current setup will make it easier to fully switch to the checkpointer when it is fixed.

        :param epoch:
        :param logs:
        :return:
        """
        # TODO: Need to get this to save only for improved loss. Like the checkpointer itself.
        with open(self.save_location, 'wb') as f:
            pickle.dump(self.working_model_as_json, f)
