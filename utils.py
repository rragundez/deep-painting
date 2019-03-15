from keras.constraints import Constraint
from keras import backend as K
import numpy as np
import cv2

from keras.layers import Conv2D, Input, Layer, Subtract, Lambda
from keras.models import Model, Sequential
from keras.applications import vgg19
from keras.optimizers import Adam

from keras import backend as K
from keras.layers import Layer

from keras.callbacks import Callback

bgr_to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
rgb_to_bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


class ShowCombination(Callback):
    def __init__(self, cv2_camera, style_img):
        self.cv2_camera = cv2_camera
        self.style_img = style_img
        self.img_height, self.img_width, _ = style_img.shape

    def on_epoch_begin(self, epoch, logs={}):
        self.content_img = cv2.resize(
            self.cv2_camera.read()[1], (self.img_width, self.img_height)
        )
        self.model.get_layer('combination_tsr').set_weights(
            [np.squeeze(preprocess_image_array(self.content_img))]
        )

    def on_epoch_end(self, epoch, logs={}):
        combination_img = self.model.get_layer('combination_tsr').get_weights()
        combination_img = deprocess_image_array(combination_img[0])
        _img = np.hstack((self.content_img, combination_img, self.style_img))
        cv2.imshow('frame', _img)
        cv2.waitKey(50)


class WeightsIdentity(Layer):

    def __init__(self, kernel_initializer='glorot_uniform', kernel_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer=self.kernel_initializer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel = K.reshape(self.kernel, (1,) + tuple(self.kernel.shape))
        return kernel

    def compute_output_shape(self, input_shape):
        return input_shape


def custom_mean_squared_error(_, content_diff):
    return K.mean(K.square(content_diff))


def gram_matrix(X):
    _X = K.squeeze(X, 0)
    features = K.batch_flatten(K.permute_dimensions(_X, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return K.expand_dims(gram / K.cast(K.prod(_X.shape), 'float32'), axis=0)


def preprocess_image_array(image):
    assert np.max(image) > 1, 'Pixel values should be in the 0-255 range'
    # 'RGB'->'BGR'
    # image = image[..., ::-1]
    image = np.expand_dims(image.astype('float32'), axis=0)
    # zero-center by mean pixel
    mean = [103.939, 116.779, 123.68]
    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    return image


def deprocess_image_array(image):
    # Remove zero-center by mean pixel
    mean = [103.939, 116.779, 123.68]
    image[..., 0] += mean[0]
    image[..., 1] += mean[1]
    image[..., 2] += mean[2]
    # 'BGR'->'RGB'
    # image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')
    return image


def get_model_chunk(model, from_layer, to_layer, include_from=False):
    model_chunk = Sequential()
    from_to_index = [i
                     for i, layer in enumerate(model.layers)
                     if (layer.name.startswith(from_layer) or layer.name.startswith(to_layer))]
    if not include_from:
        from_to_index[0] += 1

    for layer in model.layers[from_to_index[0]:from_to_index[1] + 1]:
        model_chunk.add(layer)

    return model_chunk

bgr_to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

def get_style_gram_matrices(image):
    input_img = Input(shape=image.shape)
    vgg_model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=input_img)
    style_tsr_1_gram = Lambda(gram_matrix)(vgg_model.get_layer('block1_conv1').output)
    style_tsr_2_gram = Lambda(gram_matrix)(vgg_model.get_layer('block2_conv1').output)
    style_tsr_3_gram = Lambda(gram_matrix)(vgg_model.get_layer('block3_conv1').output)
    style_tsr_4_gram = Lambda(gram_matrix)(vgg_model.get_layer('block4_conv1').output)
    style_tsr_5_gram = Lambda(gram_matrix)(vgg_model.get_layer('block5_conv1').output)
    model = Model(
        inputs=input_img,
        output=[
            style_tsr_1_gram,
            style_tsr_2_gram,
            style_tsr_3_gram,
            style_tsr_4_gram,
            style_tsr_5_gram
    ])
    return model.predict_on_batch(preprocess_image_array(image))


class WeightClip(Constraint):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)


def get_model(content_img,
              learning_rate=0.001,
              decay_rate=0.0,
              clip_weights=False):
    vgg_model = vgg19.VGG19(weights='imagenet', include_top=False)
    StyleExtractor1 = get_model_chunk(vgg_model, 'input', 'block1_conv1')
    StyleExtractor2 = get_model_chunk(vgg_model, 'block1_conv1', 'block2_conv1')
    StyleExtractor3 = get_model_chunk(vgg_model, 'block2_conv1', 'block3_conv1')
    StyleExtractor4 = get_model_chunk(vgg_model, 'block3_conv1', 'block4_conv1')
    StyleExtractor5 = get_model_chunk(vgg_model, 'block4_conv1', 'block5_conv1')

    input_img = Input(shape=content_img.shape)
    combination_tsr = WeightsIdentity(
        name="combination_tsr",
        kernel_constraint=WeightClip(min_value=-123.68, max_value=255 - 123.68) if clip_weights else None
    )(input_img)
    combination_tsr_style_1 = StyleExtractor1(combination_tsr)
    combination_tsr_style_2 = StyleExtractor2(combination_tsr_style_1)
    combination_tsr_style_3 = StyleExtractor3(combination_tsr_style_2)
    combination_tsr_style_4 = StyleExtractor4(combination_tsr_style_3)
    combination_tsr_style_5 = StyleExtractor5(combination_tsr_style_4)
    combination_tsr_style_1_gram = Lambda(gram_matrix, name='combination_tsr_style_1_gram')(combination_tsr_style_1)
    combination_tsr_style_2_gram = Lambda(gram_matrix, name='combination_tsr_style_2_gram')(combination_tsr_style_2)
    combination_tsr_style_3_gram = Lambda(gram_matrix, name='combination_tsr_style_3_gram')(combination_tsr_style_3)
    combination_tsr_style_4_gram = Lambda(gram_matrix, name='combination_tsr_style_4_gram')(combination_tsr_style_4)
    combination_tsr_style_5_gram = Lambda(gram_matrix, name='combination_tsr_style_5_gram')(combination_tsr_style_5)
    model = Model(
        inputs=input_img,
        output=[
            combination_tsr_style_1_gram,
            combination_tsr_style_2_gram,
            combination_tsr_style_3_gram,
            combination_tsr_style_4_gram,
            combination_tsr_style_5_gram
    ])
    for layer in model.layers:
        if layer.name != 'combination_tsr':
            layer.trainable = False

    losses = {
        'combination_tsr_style_1_gram': 'mean_squared_error',
        'combination_tsr_style_2_gram': 'mean_squared_error',
        'combination_tsr_style_3_gram': 'mean_squared_error',
        'combination_tsr_style_4_gram': 'mean_squared_error',
        'combination_tsr_style_5_gram': 'mean_squared_error'
    }

    model.compile(Adam(lr=learning_rate, decay=decay_rate), loss=losses)
    return model
