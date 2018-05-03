import numpy as np
import cv2
from keras import backend as K
from utils import *
from model import *
import argparse

#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * 0.9

    #Convert to row major format for using opencv routines
    if K.image_data_format() == 'channels_first':
        grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
        img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))
    else:
        grads_row_major = grads_value[0, :]
        img_row_major = gradient_ascent_step[0, :]       

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]

    #The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    if K.image_data_format() == 'channels_first':
        #Convert image back to 1 x 3 x height x width
        img = np.float32([np.transpose(img, (2, 0, 1))])
    else:
        img = np.float32([img])

    return img

def visualize_filter(input_img, filter_index, img_placeholder, layer, number_of_iterations = 20):
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer[:, filter_index, :, :])
    else:
        loss = K.mean(layer[:, :, :, filter_index])
        
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([img_placeholder], [loss, grads])

    img = input_img * 1

    # we run gradient ascent for 20 steps
    for i in range(number_of_iterations):
        img = gradient_ascent_iteration(iterate, img)

    # decode the resulting input image
    img = deprocess_image(img[0])
    print("Done with filter", filter_index)
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type = int, default = 20, help = 'Number of gradient ascent iterations')
    parser.add_argument("--img", type = str, help = \
        'Path to image to project filter on, like in google dream. If not specified, uses a random init')
    parser.add_argument("--weights_path", type = str, default = 'vgg16_weights.h5', help = 'Path to network weights file')
    parser.add_argument("--layer", type = str, default = 'conv5_1', help = 'Name of layer to use. Uses layer names in model.py')
    parser.add_argument("--num_filters", type = int, default = 16, help = 'Number of filters to vizualize, starting from filter number 0.')
    parser.add_argument("--size", type = int, default = 128, help = 'Image width and height')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)

    #Configuration:
    img_width, img_height = args.size, args.size
    filter_indexes = range(0, args.num_filters)

    #input_placeholder = K.placeholder((1, 3, img_width, img_height))
    #first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
    #first_layer.input = input_placeholder
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = get_model(input_shape)
    #model = load_model_weights(model, args.weights_path)
    model.load_weights(args.weights_path)
    layer = get_output_layer(model, args.layer)
    input_placeholder = model.input

    if args.img is None:
        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            init_img = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
        else:
            init_img = np.random.random((1, img_width, img_height, 3)) * 20 + 128.            
    else:
        img = cv2.imread(args.img, 1)
        img = cv2.resize(img, (img_width, img_height))
        if K.image_data_format() == 'channels_first':
            init_img = [np.transpose(img, (2, 0, 1))]
        else:
            init_img = [img]

    vizualizations = [None] * len(filter_indexes)
    for i, index in enumerate(filter_indexes):
        vizualizations[i] = visualize_filter(init_img, index, input_placeholder, layer, args.iterations)
        #Save the visualizations see the progress made so far
        save_filters(vizualizations, img_width, img_height)
