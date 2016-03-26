import numpy as np
import cv2
from keras import backend as K

from utils import *
from model import *

#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_grad_regularization(img, grads, percentile = 50):
    clipped = img
    gradients_columnstacked = np.reshape(grads, (grads.shape[0] * grads.shape[1], grads.shape[2]))
    clipped[np.where(grads <= np.percentile(gradients_columnstacked, 50, axis = 0))] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * 0.9

    #Convert to row major format for using opencv routines
    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_grad_regularization]

    #The reguarlization weights
    weights = np.float32([2, 2, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    #Convert image back to 1 x 3 x height x width
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

    vizualizations = [None] * len(filter_indexes)
    for i, index in enumerate(filter_indexes):
        vizualizations[i] = visualize_filter(index, input_placeholder)
        #Save the visualizations made so far to see the progress:
        save_filters(vizualizations, img_width, img_height)