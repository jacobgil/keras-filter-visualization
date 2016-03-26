## Keras CNN filter visualization utility ##


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
