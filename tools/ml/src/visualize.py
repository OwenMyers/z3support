import logging
import pickle
from matplotlib.colors import ListedColormap
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import random
from matplotlib import pyplot as plt
import numpy as np
from tools.ml import tf_vae
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

    def plot_feature_maps(self, activations, layer_names, images_per_row):
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

    def plot_weights(self, autoencoder, layer_names, images_per_row, model_is_split=False):
        if model_is_split:
            raise ValueError("This method (plot_weights) can not currently handle a split model")
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

    def plot_input_and_output(self, autoencoder, x_test, model_hash_name, model_is_split=False):
        in_out_dir = os.path.join(self.figures_project_dir, f'{model_hash_name}_in_out_images')
        if not os.path.exists(in_out_dir):
            os.mkdir(in_out_dir)
        # You will get double the number specified in the range. 10 makes 20 images
        x_test_iter = iter(x_test)
        for i in range(10):
            print(f"Plotting input and output images {i}")
            x1 = next(x_test_iter)
            x2 = next(x_test_iter)
            x = np.array([x1, x2])
            if model_is_split:
                mean, logvar = tf_vae.encode(autoencoder, x=x)
                z_points = tf_vae.reparameterize(mean=mean, logvar=logvar)
                y = tf_vae.decode(autoencoder, z_points, apply_sigmoid=True)
            else:
                y = autoencoder.predict(x)

            gl_vae_r_loss = tf_vae.gl_vae_r_loss(x, y)

            im1 = plt.imshow(x[0], aspect='auto', cmap='viridis')
            plt.colorbar(im1)
            plt.savefig(os.path.join(in_out_dir, f'{i}_0_x_example_in.png'))
            plt.clf()
            im2 = plt.imshow(y[0], aspect='auto', cmap='viridis')
            plt.colorbar(im2)
            plt.text(0, 0, f"GL VAE R Loss {gl_vae_r_loss[0]}")
            plt.savefig(os.path.join(in_out_dir, f'{i}_0_y_example_out.png'))
            plt.clf()

            im1 = plt.imshow(x[1], aspect='auto', cmap='viridis')
            plt.colorbar(im1)
            plt.savefig(os.path.join(in_out_dir, f'{i}_1_x_example_in.png'))
            plt.clf()
            im2 = plt.imshow(y[1], aspect='auto', cmap='viridis')
            plt.text(0, 0, f"GL VAE R Loss {gl_vae_r_loss[1]}")
            plt.colorbar(im2)
            plt.savefig(os.path.join(in_out_dir, f'{i}_1_y_example_out.png'))
            plt.clf()

    def plot_decoder_result_from_input(self, autoencoder, start_loc=(-1, -1), end_loc=(1, 1), layer_names=None,
                                       model_is_split=False):
        # Trying to do this using suggestion:
        # https://stackoverflow.com/questions/49193510/how-to-split-a-model-trained-in-keras
        decoder = None
        if not model_is_split:
            decoder_input = None
            started = False
            for i, layer_name in enumerate(layer_names):
                if started:
                    if i+1 > len(layer_names)-1:
                        break
                    current_layer = autoencoder.layers[i+1]
                    decoder = current_layer(decoder)
                elif layer_name == 'dense_encoder_output':
                    started = True
                    print("hey hey")
                    decoder_input = Input(autoencoder.layers[i+1].input_shape[1:])
                    decoder = decoder_input
                    decoder = autoencoder.layers[i + 1](decoder)
            decoder = Model(inputs=decoder_input, outputs=decoder)

        # create the path that that we want to cut across
        num_steps = 5
        loc_list = []
        x_step_size = (end_loc[0] - start_loc[0])/num_steps
        y_step_size = (end_loc[1] - start_loc[1])/num_steps
        for i in range(num_steps):
            current_loc = [None, None]
            current_loc[0] = start_loc[0] + i * x_step_size
            current_loc[1] = start_loc[1] + i * y_step_size
            loc_list.append(current_loc)
        if model_is_split:
            results = tf_vae.decode(autoencoder, np.matrix(loc_list), apply_sigmoid=True)
        else:
            results = decoder.predict(loc_list)
        if not os.path.exists(os.path.join(self.figures_project_dir, 'latent_slice_video')):
            os.mkdir(os.path.join(self.figures_project_dir, 'latent_slice_video'))
        for index, cur_result in enumerate(results):
            plt.imshow(cur_result, aspect='auto', cmap='viridis')
            plt.savefig(os.path.join(self.figures_project_dir, 'latent_slice_video', f'slice_{index}.png'))
            plt.clf()

    def main(self, model_path):
        if self.use_current_checkpoint:
            model = self.get_checkpoint_model(model_path)
        else:
            model = tf.keras.models.load_model(model_path, custom_objects={'compute_loss': tf_vae.compute_loss})

        model_hash_name = model_path.strip('/').split('/')[-1].split('.')[0]
        x_test = self.get_testing_data()
        y_test = self.get_testing_data_labels()

        # self.plot_feature_maps(autoencoder, activations, x_test, encoder_layer_names, images_per_row)
        # self.plot_weights(autoencoder, encoder_layer_names, images_per_row)
        self.plot_decoder_result_from_input(model, start_loc=[-1.0, 0.0], end_loc=[2.0, 0.0], model_is_split=True)
        #self.plot_decoder_result_from_input(model, start_loc=[1.0, 1.5], end_loc=[-1.0, -1.5], model_is_split=True)
        self.simple_plot_dense_layer(model, model_hash_name, x_test, y_test)
        self.plot_input_and_output(model, x_test, model_hash_name, model_is_split=True)

    def old_plot_dense_layer(self, autoencoder, layer_names, activations, labels):
        for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
            if layer_name == 'dense_encoder_output':
                plt.scatter(layer_activation[:, 0], layer_activation[:, 1], c=labels, cmap='Set1')
                plt.show()
                plt.savefig(os.path.join(self.figures_project_dir, 'dense_layer.png'))

    def simple_plot_dense_layer(self, model, model_hash_name, x_test, y_test):
        x_in = self.get_testing_data()
        y_in = self.get_testing_data_labels()

        mean, logvar = tf_vae.encode(model, x=x_in)
        z = tf_vae.reparameterize(mean=mean, logvar=logvar)
        if isinstance(y_in[0], str):
            unique_lables = {i for i in y_in}
            number_mapping = {}
            for i, j in enumerate(unique_lables):
                number_mapping[j] = i
            c_arr = []
            for i in y_in:
                c_arr.append(number_mapping[i])
        else:
            c_arr = y_in

        colours = ListedColormap(['r', 'b', 'g'])
        #plt.set_cmap('viridis')
        scatter = plt.scatter(z[:, 0], z[:, 1], c=c_arr, s=1, cmap=colours)
        plt.legend(handles=scatter.legend_elements()[0], labels=[1, 2, 3])
        plt.savefig(os.path.join(self.figures_project_dir, f'{model_hash_name}_dense_layer.png'))
        plt.clf()


def gdl_external_viz_in_out(path_to_model, settings_file, ignore_this_run_loc):
    from gdl_code_repeate.vae_model import VariationalAutoencoder
    from gdl_code_repeate.utils.loaders import load_model
    loaded_model = load_model(VariationalAutoencoder, path_to_model)
    viz_tool = VizTool(settings_file, ignore_this_run_loc, False)
    x_test = viz_tool.get_testing_data()
    y_test = viz_tool.get_testing_data_labels()
    viz_tool.plot_input_and_output(loaded_model, x_test[:1000], 'gdl_code_repeate/figures/', model_is_split=True)

    n_to_show = 5000
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = loaded_model.encoder.predict(example_images)

    plt.figure()
    plt.scatter(z_points[:, 0], z_points[:, 1], c=example_labels, alpha=0.5, s=2)
    plt.savefig('gdl_code_repeate/figures/vae_latent_space.png', dpi=200)

    viz_tool.plot_decoder_result_from_input(loaded_model, start_loc=[-0.75, 1.5], end_loc=[-1.5, 0.0],
                                            model_is_split=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--model_path', type=str, help='Path to model', required=True)
    parser.add_argument('--settings', type=str, help='Settings file location', required=False)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    parser.add_argument('--use-current-checkpoint', help='If this is used then the current checkpoint in'
                                                         'model_checkpoints will be used instead of the best model.'
                                                         'This is to be used for evaluating models mid run.',
                                                         action='store_true')
    parser.add_argument('--gdl_external_model', type=str, help='Path to model', required=False, default=None)
    parser.add_argument('--gdl_external_type', type=str, help='Specifies if the saved model is a tf format or h5',
                        required=False, default='tf')

    args = parser.parse_args()

    if args.gdl_external_model is not None:
        gdl_external_viz_in_out(args.gdl_external_model, args.settings, args.run_location)
    else:
        tool = VizTool(args.settings, args.run_location, args.use_current_checkpoint)
        tool.main(model_path=args.model_path)
