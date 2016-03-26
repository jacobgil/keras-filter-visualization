import numpy as np
import cv2
import h5py
from keras import backend as K

from utils import *
from model import *

def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])
    
    gradient_ascent_step = img + grads_value * 0.9
    #return gradient_ascent_step
    transposed_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    decay = 0.8 * (transposed_row_major)
    blur = cv2.blur(transposed_row_major, (3, 3))
    
    clip_weak_gradients = transposed_row_major
    grads_value_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    gradients_columnstacked = np.reshape(grads_value_row_major, \
                              (grads_value_row_major.shape[0] * grads_value_row_major.shape[1], \
                              grads_value_row_major.shape[2]))
    
    clip_weak_gradients[np.where(grads_value_row_major <= np.percentile(gradients_columnstacked, 50, axis = 0))] = 0
    
    weights = np.float32([0, 5, 0])
    weights /= np.sum(weights)

    img = weights[0] * decay +  weights[1] * blur + weights[2] * clip_weak_gradients

    img = np.float32([np.transpose(img, (2, 0, 1))])
    return img

def visualize_filter(filter_index, img_placeholder):
    loss = K.mean(layer[:, filter_index, :, :])
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([img_placeholder], [loss, grads])

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
    # we run gradient ascent for 20 steps
    for i in range(20):
        input_img_data = gradient_ascent_iteration(iterate, input_img_data)

    # decode the resulting input image
    img = deprocess_image(input_img_data[0])
    print "Done with filter", filter_index
    return img

if __name__ == "__main__":

    #Configuration:
    img_width, img_height = 128, 128
    weights_path = 'vgg16_weights.h5'
    layer_name = 'conv5_1'
    filter_indexes = range(50, 50 + 16)

    input_placeholder = K.placeholder((1, 3, img_width, img_height))
    first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
    first_layer.input = input_placeholder


    model = get_model(first_layer)
    model = load_model_weights(model, weights_path)
    layer = get_output_layer(model, layer_name)

    vizualizations = [visualize_filter(index, input_placeholder) for index in filter_indexes]
    save_filters(vizualizations)