import logging
from matplotlib import pyplot as plt
import numpy as np
from tools.ml.base import MLToolMixin
import argparse
import os


class VizTool(MLToolMixin):
    def __init__(self, settings_file, working_location):
        super().__init__(settings_file, working_location)
        assert os.path.exists(os.path.join(working_location, 'figures'))

        self.figures_project_dir = os.path.join(working_location, 'figures', self.timestamp)
        if not os.path.exists(self.figures_project_dir):
            os.mkdir(self.figures_project_dir)

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

    def main(self):
        autoencoder = self.get_best_autoencoder()
        activation_model = self.get_best_activations()

        activations = activation_model.predict(x_test)
        images_per_row = 16
        layer_names = []
        layers_to_encoded = int(len(autoencoder.layers) / 2)
        for layer in autoencoder.layers[:layers_to_encoded]:
            layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            if 'input' in layer_name:
                continue
            size, n_cols, grid_dimensions = self.get_current_layer_display_grid_size(images_per_row, layer_activation)
            display_grid = np.zeros(grid_dimensions)
            for col in range(n_cols):  # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    try:
                        channel_image = layer_activation[0, :, :, col * images_per_row + row]
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
        plt.savefig(os.path.join(self.figures_project_dir, 'feature_map.png'))

        neuron = 0
        for layer_name in layer_names:
            print(layer_name)
            if 'conv2d' not in layer_name:
                continue

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
        plt.savefig(os.path.join(self.figures_project_dir, 'layer_weights.png'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    args = parser.parse_args()

    tool = VizTool(args.settings, args.run_location)
    tool.main()
