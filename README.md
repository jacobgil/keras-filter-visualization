## Keras CNN filter visualization utility ##

This is a utility for visualizing convolution filters in a Keras CNN model.
Check [this](http://jacobcv.blogspot.com/2016/03/visualizing-cnn-filters-with-keras_26.html) blog post.

By default this uses VGG16.  Get the reduced model without the fully connected layers from here:
https://github.com/awentzonline/keras-vgg-buddy

You can use the utility to project filters on a random image initial image, or on your own image to produce deep-dream like results.

This is quite compute intensive and can take a few minutes depending on image sizes and number of filters.
An intermediate image is written to disk so you can see the progress done so far.

----------

    usage: viz.py [-h] [--iterations ITERATIONS] [--img IMG]
              [--weights_path WEIGHTS_PATH] [--layer LAYER]
              [--num_filters NUM_FILTERS] [--size SIZE]

    optional arguments:
      -h, --help            show this help message and exit
      --iterations ITERATIONS
                            Number of gradient ascent iterations
      --img IMG             Path to image to project filter on, like in google
                            dream. If not specified, uses a random init
      --weights_path WEIGHTS_PATH
                            Path to network weights file
      --layer LAYER         Name of layer to use. Uses layer names in model.py
      --num_filters NUM_FILTERS
                            Number of filters to vizualize, starting from filter
                            number 0.
      --size SIZE           Image width and height

![256 filters from VGG16](https://github.com/jacobgil/keras-filter-visualization/blob/master/examples/10x10.png?raw=true)
