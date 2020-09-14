from matplotlib import pyplot as plt
import numpy as np


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


def fill_display_grid(to_fill, cur_image, cur_col, cur_row, cur_size):
    """Will modify data in reference ``to_fill``"""
    cur_image -= cur_image.mean() # Post-processes the feature to make it visually palatable
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
            cur_col * cur_size: (cur_col + 1) * cur_size, # Displays the grid
            cur_row * cur_size: (cur_row + 1) * cur_size
        ] = cur_image
    except ValueError:
        to_fill[
            cur_col * cur_size: (cur_col + 1) * cur_size, # Displays the grid
            cur_row * cur_size: (cur_row + 1) * cur_size
        ] = cur_image


def get_plottable_weights(raw_weights_from_get_layer):
    # short name
    w = raw_weights_from_get_layer
    print('in w.shape')
    print(w.shape)
    if len(w.shape) != 4:
        raise ValueError('Unexpected shape of weights')
    #w = w.reshape((w.shape[0], w.shape[1], w.shape[2]*w.shape[3]))
    #print('shape after reshape')
    #print(w.shape)
    return w


def main():

    images_per_row = 16
    layer_names = []
    for layer in autoencoder.layers[:layers_to_encoded]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        if 'input' in layer_name:
            continue
        size, n_cols, grid_dimensions = get_current_layer_display_grid_size(images_per_row, layer_activation)
        display_grid = np.zeros(grid_dimensions)
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                try:
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                except IndexError:
                    continue
                fill_display_grid(display_grid, channel_image, col, row, size)
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


    neuron = 0
    for layer_name in layer_names:
        print(layer_name)
        if 'conv2d' not in layer_name:
            continue

        weights = autoencoder.get_layer(name=layer_name).get_weights()[0]
        for_plot = get_plottable_weights(weights)

        size, n_cols, grid_dimensions = get_current_layer_display_grid_size(images_per_row, for_plot)
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
                fill_display_grid(display_grid, channel_image, col, row, size)

        scale = 1.0 / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


if __name__ == "__main__":
    main()