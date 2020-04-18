from sklearn.metrics.classification import accuracy_score
from sklearn.cross_decomposition import PLSRegression
from keras.layers import *
from keras.models import Model
import random

import keras
from keras.callbacks import Callback
class LearningRateScheduler(Callback):

    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        lr = self.init_lr
        for i in range(0, len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print('Learning rate:{}'.format(lr))
        #K.set_value(self.model.optimizer.lr, lr)
        keras.backend.set_value(self.model.optimizer.lr, lr)

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

def random_crop(img=None, random_crop_size=(64, 64)):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def data_augmentation(X, padding=4):

    X_out = np.zeros(X.shape, dtype=X.dtype)
    n_samples, x, y, _ = X.shape

    padded_sample = np.zeros((x+padding*2, y+padding*2, 3), dtype=X.dtype)

    for i in range(0, n_samples):
        p = random.random()
        padded_sample[padding:x+padding, padding:y+padding, :] = X[i][:, :, :]
        if p >= 0.5: #random crop on the original image
            X_out[i] = random_crop(padded_sample, (x, y))
        else: #random crop on the flipped image
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))

    return X_out

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

def vip(model):
    t = model.x_scores_  # (n_samples, n_components)
    w = model.x_weights_  # (p, n_components)
    q = model.y_loadings_  # (q, n_components)
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    w_norm = np.linalg.norm(w, axis=0)
    weights = (w / np.expand_dims(w_norm, axis=0)) ** 2
    return np.sqrt(p * (weights @ s).ravel() / np.sum(s))

def score(model=None, X_train=None, y_train=None, n_components=2, layers=[]):
    # Extract the features
    outputs = []

    for i in layers:
        layer = model.get_layer(index=i).output
        outputs.append(Flatten()(AveragePooling2D(pool_size=8, name='avg{}_feature'.format(i))(layer)))

    model = Model(model.input, outputs)
    X_train = model.predict(X_train)

    if len(layers) == 1:
        X_train = [X_train]
    ranked = []

    for i in range(0, len(layers)):
        dm = PLSRegression(n_components=n_components)
        dm.fit(X_train[i], y_train)
        scores = vip(dm)

        ranked.append(np.mean(scores))

    #print(ranked)
    return ranked

def insert_head(cnn_model, idx_stop):

    H = cnn_model.get_layer(index=idx_stop).output

    H = Activation('relu', name='Logits')(H)
    H = AveragePooling2D(pool_size=8)(H)
    H = Flatten()(H)
    H = Dense(10, activation='softmax', kernel_initializer='he_normal')(H)

    return Model(cnn_model.input, H)

if __name__ == '__main__':
    np.random.seed(12227)

    debug = False
    n_components = 4

    block = 2  # Only required by stopImportance criterion

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    lr = 0.01
    schedule = [(100, 1e-3), (150, 1e-4)]

    #idx_blocks = [0, 67, 131] for ResNet56,  idx_blocks = [0, 130, 257] for ResNet110
    idx_blocks = [0, 25, 47]

    cnn_model = load_model(architecture_file='ResNet20', weights_file='ResNet20')
    n_params_unpruned = cnn_model.count_params()
    flops_unpruned, _ = compute_flops(cnn_model)
    acc_unpruned = accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnn_model.predict(x_test), axis=1))

    lr_scheduler = LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    all_add = []
    for i in range(idx_blocks[block], len(cnn_model.layers)):
        if isinstance(cnn_model.get_layer(index=i), Add):
            all_add.append(i)

    score_all = score(cnn_model, x_train, y_train,
                      n_components=n_components,
                      layers=all_add)

    idx_stop = 0
    for i in range(0, len(score_all) - 1):
        if score_all[i + 1] > score_all[i]:
            idx_stop = i + 1
        else:
            idx_stop = all_add[idx_stop]
            break

    cnn_model = insert_head(cnn_model, idx_stop)

    sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

    for ep in range(1, 1):

        x_tmp = np.concatenate((data_augmentation(x_train),
                                data_augmentation(x_train),
                                data_augmentation(x_train)))
        y_tmp = np.concatenate((y_train,
                                y_train,
                                y_train))

        cnn_model.fit(x_tmp, y_tmp, batch_size=128,
                      callbacks=callbacks, verbose=2,
                      epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0:
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnn_model.predict(x_test), axis=1))
            print('Accuracy [{:.4f}]'.format(acc))


    y_pred = cnn_model.predict(x_test)
    acc_pruned = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    n_params_pruned = cnn_model.count_params()
    flops_pruned, _ = compute_flops(cnn_model)

    print('Original (Unpruned) Network. Number of Parameters [{}] FLOPS [{}] Accuracy [{:.4f}]'
          .format(n_params_unpruned, flops_unpruned, acc_unpruned))
    print('Pruned Network. Number of Parameters [{}] FLOPS [{}] Accuracy [{:.4f}] Pruned at Layer [{}]'
          .format(n_params_pruned, flops_pruned, acc_pruned, idx_stop))