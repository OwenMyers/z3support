import logging
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import random
from matplotlib import pyplot as plt
import numpy as np
from tools.ml.src.base import MLToolMixin
import argparse
import os


class VizTool(MLToolMixin):
    def __init__(self, settings_file, working_location, use_current_checkpoint):
        super().__init__(settings_file, working_location)
        assert os.path.exists(os.path.join(working_location, 'figures'))
        self.n_feature_maps = int(self.config['Plotting']['N_FEATURE_MAPS'])

        self.figures_project_dir = os.path.join(working_location, 'figures', self.timestamp)
        if not os.path.exists(self.figures_project_dir):
            os.mkdir(self.figures_project_dir)

        self.use_current_checkpoint = use_current_checkpoint

    @staticmethod
    def get_current_layer_display_grid_size(images_per_row, matrix_in):
        print(f'Layer shape {matrix_in.shape}')
        # Number of features in the feature map
        n_features = matrix_in.shape[-1]
        print(f'n_features {n_features}')
        # The feature map has shape (1, size, size, n_features).
        size = matrix_in.shape[1]
        print(f'size {size}')
        # Tiles the activation channels in this matrix
        n_cols = max([n_features // images_per_row, 1])
        print(f'n_cols {n_cols}')

        return size, n_cols, (size * n_cols, min(images_per_row, n_features) * size)

    @staticmethod
    def fill_display_grid(to_fill, cur_image, cur_col, cur_row, cur_size):
        """Will modify data in reference ``to_fill``"""
        cur_image -= cur_image.mean()  # Post-processes the feature to make it visually palatable
        cur_image /= cur_image.std()
        cur_image *= 64
        cur_image += 128
        cur_image = np.clip(cur_image, 0, 255).astype('uint8')
        print(f'cur_col * cur_size {cur_col * cur_size}')
        print(f'(cur_col + 1) * cur_size {(cur_col + 1) * cur_size}')
        print(f'cur_row * cur_size {cur_row * cur_size }')
        print(f'(cur_row + 1) * cur_size {(cur_row + 1) * cur_size}')
        print(f'cur_image.shape {cur_image.shape}')
        try:
            to_fill[
                cur_col * cur_size: (cur_col + 1) * cur_size,  # Displays the grid
                cur_row * cur_size: (cur_row + 1) * cur_size
            ] = cur_image
        except ValueError:
            to_fill[
                cur_col * cur_size: (cur_col + 1) * cur_size,  # Displays the grid
                cur_row * cur_size: (cur_row + 1) * cur_size
            ] = cur_image

    @staticmethod
    def get_plottable_weights(raw_weights_from_get_layer):
        # short name
        w = raw_weights_from_get_layer
        print('in w.shape')
        print(w.shape)
        if len(w.shape) != 4:
            raise ValueError('Unexpected shape of weights')
        # w = w.reshape((w.shape[0], w.shape[1], w.shape[2]*w.shape[3]))
        # print('shape after reshape')
        # print(w.shape)
        return w

    def plot_feature_maps(self, autoencoder, activations, x_test, layer_names, images_per_row):
        # Note: the `autoencoder` parameter is not currently used. We have it in here because it is required
        # for the solution we are trying to build to plot information about checkpointed models, which requires
        # information about the structure of the model being run. There is a known problem with the TF
        # checkpointers, that you can not save the weights with the structure. It is an option but known not
        # to work.
        autoencoder = None
        if autoencoder is None:
            logging.debug("Autoencoder unused in plot_feature_maps method.")

        # List of the indices of the rows of data that will be used to display feature maps
        feature_list = []
        while len(feature_list) < self.n_feature_maps:
            #current_feature = random.randint(0, len(x_test) - 1)
            current_feature = random.randint(0, 100)
            if current_feature in feature_list:
                continue
            feature_list.append(current_feature)
            for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
                if 'input' in layer_name:
                    continue
                size, n_cols, grid_dimensions = self.get_current_layer_display_grid_size(images_per_row,
                                                                                         layer_activation)
                display_grid = np.zeros(grid_dimensions)
                for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                    for row in range(images_per_row):
                        try:
                            channel_image = layer_activation[current_feature, :, :, col * images_per_row + row]
                        except IndexError:
                            continue
                        self.fill_display_grid(display_grid, channel_image, col, row, size)
                scale = 1. / size
                plt.figure(figsize=(scale * display_grid.shape[1],
                                    scale * display_grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                # noinspection SpellCheckingInspection
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.savefig(os.path.join(self.figures_project_dir,
                                         layer_name + f'feature_map_{current_feature}.png'))

    def plot_weights(self, autoencoder, layer_names, images_per_row):
        neuron = 0
        for layer_name in layer_names:
            print(layer_name)
            if 'conv2d' not in layer_name:
                continue

            # get_weights returns a list of length 2
            #   0 -> weights
            #   1 -> bias
            weights = autoencoder.get_layer(name=layer_name).get_weights()[0]
            for_plot = self.get_plottable_weights(weights)

            size, n_cols, grid_dimensions = self.get_current_layer_display_grid_size(images_per_row, for_plot)
            display_grid = np.zeros(grid_dimensions)
            print(f'display_grid shape {display_grid.shape}')
            for col in range(n_cols):
                for row in range(images_per_row):
                    try:
                        channel_image = for_plot[:, :, neuron, col * images_per_row + row]
                    except IndexError:
                        continue
                    print(f'col {col}')
                    print(f'row {row}')
                    self.fill_display_grid(display_grid, channel_image, col, row, size)

            scale = 1.0 / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            # noinspection SpellCheckingInspection
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig(os.path.join(self.figures_project_dir, layer_name + 'layer_weights.png'))

    def plot_input_and_output(self, autoencoder, x_test):
        x1 = next(iter(x_test))
        x2 = next(iter(x_test))
        x = np.array([x1, x2])
        y = autoencoder.predict(x)
        #y = np.array(y[:,:,0,:])
        plt.imshow(x[0], aspect='auto', cmap='viridis')
        plt.savefig(os.path.join(self.figures_project_dir, 'example_in.png'))
        plt.imshow(y[0], aspect='auto', cmap='viridis')
        plt.savefig(os.path.join(self.figures_project_dir, 'example_out.png'))

    def plot_dense_layer(self, autoencoder, layer_names, activations, labels):
        #weights = autoencoder.get_layer(name='dense_encoder_output').get_weights()
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            if layer_name == 'dense_encoder_output':
                print("hi")
                plt.scatter(layer_activation[:, 0], layer_activation[:, 1], c=labels, cmap='Set1')#, s=1)
                plt.show()
                plt.savefig(os.path.join(self.figures_project_dir, 'dense_layer.png'))
                #plt.xlim()

    def plot_decoder_result_from_input(self, autoencoder, layer_names, input):
        # Trying to do this using suggestion https://stackoverflow.com/questions/49193510/how-to-split-a-model-trained-in-keras
        started = False
        for i, layer_name in enumerate(layer_names):  # Displays the feature maps
            if started:
                decoder = autoencoder.layers[i+1](decoder)
            elif layer_name == 'dense_encoder_output':
                started = True
                print("hey hey")
                decoder_input = Input(autoencoder.layers[i+1].input_shape[1:])
                decoder = decoder_input
        decoder = Model(inputs=decoder_input, outputs=decoder)


    def main(self):
        if self.use_current_checkpoint:
            autoencoder = self.get_checkpoint_model()
        else:
            autoencoder = self.get_best_autoencoder()
        activation_model = self.get_best_activations()

        x_test = self.get_testing_data()
        y_test = self.get_testing_data_labels()

        #activation_model.predict(np.zeros([5, 28, 28, 1]))
        x_test = x_test[:1000]
        y_test = y_test[:1000]
        #input_for_act = list(x_test.as_numpy_iterator())
        #input_for_act = np.array(input_for_act)[:, 0, :, :]
        activations = activation_model.predict(x_test)
        images_per_row = 16
        encoder_layer_names = []
        layers_to_encoded = int(len(autoencoder.layers) / 2)
        for layer in autoencoder.layers[:layers_to_encoded+1]:
            # Names of the layers to include in plot
            encoder_layer_names.append(layer.name)

        #self.plot_feature_maps(autoencoder, activations, x_test, encoder_layer_names, images_per_row)
        #self.plot_weights(autoencoder, encoder_layer_names, images_per_row)
        #self.plot_dense_layer(autoencoder, encoder_layer_names, activations, y_test)
        #self.plot_input_and_output(autoencoder, x_test)
        self.plot_decoder_result_from_input(autoencoder, encoder_layer_names, input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    parser.add_argument('--use-current-checkpoint', help='If this is used then the current checkpoint in'
                                                         'model_checkpoints will be used instead of the best model.'
                                                         'This is to be used for evaluating models mid run.',
                                                         action='store_true')
    args = parser.parse_args()

    tool = VizTool(args.settings, args.run_location, args.use_current_checkpoint)
    tool.main()
