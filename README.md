# Keras implementation of [PSPNet(caffe)](https://github.com/hszhao/PSPNet)

Pyramid Scene Parsing Network (PSPNet) was proposed by Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Details are in [project page](https://hszhao.github.io/projects/pspnet/index.html).

'[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)' was ranked 1st place in [ImageNet Scene Parsing Challenge 2016](http://image-net.org/challenges/LSVRC/2016/results). The code is modified from Caffe version of [yjxiong](https://github.com/yjxiong/caffe/tree/mem) and [DeepLab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for evaluation. We merge the batch normalization layer named 'bn_layer' in the former one into the later one while keep the original 'batch_norm_layer' in the later one unchanged for compatibility. The difference is that 'bn_layer' contains four parameters as 'slope,bias,mean,variance' while 'batch_norm_layer' contains two parameters as 'mean,variance'. Several evaluation code is borrowed from [MIT Scene Parsing](https://github.com/CSAILVision/sceneparsing).

The architecture is described as follows:
![New](assets/pspnet.png)
Figure 1. Overview of our proposed PSPNet. Given an input image (a), we first use CNN to get the feature map of the last convolutional layer (b), then a pyramid parsing module is applied to harvest different sub-region representations, followed by upsampling and concatenation layers to form the final feature representation, which carries both local and global context information in (c). Finally, the representation is fed into a convolution layer to get the final per-pixel prediction (d).

This is an implementation of PSPNet in Keras.

### Setup
1. Install dependencies:
    * Tensorflow (-gpu)
    * Keras
    * numpy
    * scipy
    * pycaffe(PSPNet)(optional for converting the weights) 
    ```bash
    pip install -r requirements.txt --upgrade
    ```
2. Converted trained weights are needed to run the network.
Weights(in ```.h5 .json``` format) have to be downloaded and placed into directory ``` weights/keras ```


Already converted weights can be downloaded here:

 * [pspnet50_ade20k.h5](https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1)
[pspnet50_ade20k.json](https://www.dropbox.com/s/v41lvku2lx7lh6m/pspnet50_ade20k.json?dl=1)
 * [pspnet101_cityscapes.h5](https://www.dropbox.com/s/c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1)
[pspnet101_cityscapes.json](https://www.dropbox.com/s/fswowe8e3o14tdm/pspnet101_cityscapes.json?dl=1)
 * [pspnet101_voc2012.h5](https://www.dropbox.com/s/uvqj2cjo4b9c5wg/pspnet101_voc2012.h5?dl=1)
[pspnet101_voc2012.json](https://www.dropbox.com/s/rr5taqu19f5fuzy/pspnet101_voc2012.json?dl=1)

Running this needs the compiled original PSPNet caffe code and pycaffe.

```bash
python weight_converter.py <path to .prototxt> <path to .caffemodel>
```

## Usage:

```bash
python pspnet.py -m <model> -i <input_image>  -o <output_path>
python pspnet.py -m pspnet101_cityscapes -i example_images/cityscapes.png -o test/cityscapes.jpg
python pspnet.py -m pspnet101_voc2012 -i example_images/pascal_voc.jpg -o test/pascal_voc.jpg
python pspnet.py -m pspnet50_ade20k -i example_images/ade20k.jpg -o test/ade20k.jpg
python pspnet.py -m pspnet101_voc2012 -i example_images/pascal_voc_2007_000733.jpg -o test/pascal_voc_2007_000733.jpg
```
List of arguments:
```bash
 -m --model        - which model to use: 'pspnet50_ade20k', 'pspnet101_cityscapes', 'pspnet101_voc2012'
    --id           - (int) GPU Device id. Default 0
 -s --sliding      - Use sliding window
 -f --flip         - Additional prediction of flipped image
 -ms --multi_scale - Predict on multiscale images
```
## Keras results:
![Original](example_images/ade20k.jpg)
![New](example_results/ade20k_seg.jpg)
![New](example_results/ade20k_seg_blended.jpg)
![New](example_results/ade20k_probs.jpg)

![Original](example_images/cityscapes.png)
![New](example_results/cityscapes_seg.jpg)
![New](example_results/cityscapes_seg_blended.jpg)
![New](example_results/cityscapes_probs.jpg)

![Original](example_images/pascal_voc.jpg)
![New](example_results/pascal_voc_seg.jpg)
![New](example_results/pascal_voc_seg_blended.jpg)
![New](example_results/pascal_voc_probs.jpg)


## Implementation details
* The interpolation layer is implemented as custom layer "Interp"
* Forward step takes about ~1 sec on single image
* Memory usage can be optimized with:
    ```python
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3 
    sess = tf.Session(config=config)
    ```
* ```ndimage.zoom``` can take a long time





