#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join
import argparse
import numpy as np

#from scipy import misc
#from skimage import io        
import imageio as io
from skimage.transform import resize # for image processing

from scipy import ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf
import layers_builder as layers
from dataset import utils
from python_utils.preprocessing import preprocess_img
from keras.utils.generic_utils import CustomObjectScope

class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape

        # These are the means for the ImageNet pretrained ResNet
        self.data_mean = np.array([[[123.68, 116.779, 103.939]]])  # RGB order

        if 'pspnet' in weights:
            # keras weights
            json_path = join("weights", "keras", weights + ".json")
            h5_path = join("weights", "keras", weights + ".h5")
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path, 'r') as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def image_balance(self, img):
        """ subtract mean value
        Arguments:
          img: image
        Returns:
          the balanced image
        """
        img = img - self.data_mean
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')

        return img

    def predict(self, oimg, flip_evaluation=False):
        """
        Predict segementation for an image.

        Arguments:
            oimg: The original input image. must be rowsxcolsx3
        """
        h_ori, w_ori = oimg.shape[:2]

        #img = misc.imresize(img, self.input_shape)

        # Preprocessing. 
        # Note: this preserve_range has to be true, or else the type is 'float64'
        img = resize(oimg, self.input_shape, preserve_range=True)

        # balance image
        img = self.image_balance(img)

        print("Predicting...")

        probs = self.feed_forward(img, flip_evaluation)

        # back to original image size
        if oimg.shape != self.input_shape:  # upscale prediction if necessary
            h, w = probs.shape[:2]
            probs = ndimage.zoom(probs, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        print("Finished prediction...")

        return probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array([data, np.flip(data, axis=1)])
            # 
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[0] 
                          + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]

        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_path', type=str, default='example_results/ade20k.jpg',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('--input_size', type=int, default=500)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():

        print(args)
        if not args.weights:
            if "pspnet50" in args.model:
                # ade20k
                pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                                  weights=args.model)
            elif "pspnet101" in args.model:
                if "cityscapes" in args.model:
                    pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                       weights=args.model)
                if "voc2012" in args.model:
                    pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                       weights=args.model)

            else:
                print("Network architecture not implemented.")
        else:
            pspnet = PSPNet50(nb_classes=2, input_shape=(768, 480), 
                                weights=args.weights)

        # original image
        #img = misc.imread(args.input_path, mode='RGB')
        img  = io.imread(args.input_path)

        # predicted 
        probs = pspnet.predict(img, args.flip)

        # post processing
        print("Writing results...")
        cm = np.argmax(probs, axis=2)
        pm = np.max(probs, axis=2)

        #color_cm = utils.add_color(cm)    # label --> rgb
        # color cm is [0.0-1.0] img is [0-255]
        #alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

        # blending original image with segmented image
        color_cm = utils.color_class_image(cm, args.model)
        alpha_blended = 0.5 * color_cm + 0.5 * img

        #
        filename, ext = splitext(args.output_path)

        # save the results
        io.imsave(filename + "_seg_read" + ext, cm)
        io.imsave(filename + "_seg" + ext, color_cm)
        io.imsave(filename + "_probs" + ext, pm)
        io.imsave(filename + "_seg_blended" + ext, alpha_blended)
