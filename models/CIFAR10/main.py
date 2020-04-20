import keras
import numpy as np
from sklearn.metrics.classification import accuracy_score

def compute_flops(model):
    import keras
    total_flops =0
    flops_per_layer = []

    try:
        layer = model.get_layer(index=1).layers #Just for discover the model type
        for layer_idx in range(1, len(model.get_layer(index=1).layers)):
            layer = model.get_layer(index=1).get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)
    except:
        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)

    return total_flops, flops_per_layer

def load_model(architecture_file='', weights_file=''):
    import keras
    from keras.utils.generic_utils import CustomObjectScope

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file))
    else:
        print('Load architecture [{}]'.format(architecture_file))

    return model

if __name__ == '__main__':
    np.random.seed(12227)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean
    y_test = keras.utils.to_categorical(y_test, 10)


    architecture_name = 'ResNet20/cifar10ResNet20_prunedLayer60'
    weights = 'ResNet20/cifar10ResNet20_prunedLayer60'


    model = load_model(architecture_name, weights)

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred)

    n_params = model.count_params()
    flops, _ = compute_flops(model)

    print('Number of Parameters [{}] FLOPS [{}] Accuracy [{:.4f}]'.format(n_params, flops, acc))