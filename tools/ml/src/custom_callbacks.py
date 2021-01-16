from keras.callbacks import Callback
import os
import pickle

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
        if not os.path.exists(os.path.join(self.save_location, 'model_checkpoints')):
            raise ValueError(f"Directory model_checkpoints must exist in {self.save_location}")
        with open(os.path.join(self.save_location, 'model_checkpoints', 'pickled_compiled_model.pkl'), 'wb') as f:
            pickle.dump(self.working_model_as_json, f)
