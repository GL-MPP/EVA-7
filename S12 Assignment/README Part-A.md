# Session - 12 Assignment (Part A)

## Requirement

- Modify existing [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10.
- Train the model for 50 epochs


## Approach

There are 2 parts to this exercise.

Part -1 Modify the network (both the spatial transformation block and the CNN block) to make it compatible with CIFAR10 dataset 32x32x3.

Part -2 Train the model for 50 epochs and generate a set of transformed images transformed by the the Spatial transformer.


### Part -1
A desirable property of a system which is able to reason about images is to disentangle object pose and part deformation from texture and shape. The introduction of local max-pooling layers in CNNs has helped to satisfy this property by allowing a network to be somewhat spatially invariant
to the position of features. However, due to the typically small spatial support for max-pooling (e.g. 2 × 2 pixels) this spatial invariance is only realised over a deep hierarchy of max-pooling and convolutions, and the intermediate feature maps (convolutional layer activations) in a CNN are not
actually invariant to large transformations of the input data. A Spatial Transformer module has been introduced, that can be included into a standard neural
network architecture to provide spatial transformation capabilities. The action of the spatial transformer is conditioned on individual data samples, with the appropriate behaviour learnt during training for the task in question (without extra supervision). Unlike pooling layers, where the receptive
fields are fixed and local, the spatial transformer module is a dynamic mechanism that can actively
spatially transform an image (or a feature map) by producing an appropriate transformation for each
input sample. The transformation is then performed on the entire feature map (non-locally) and
can include scaling, cropping, rotations, as well as non-rigid deformations. This allows networks
which include spatial transformers to not only select regions of an image that are most relevant (attention), but also to transform those regions to a canonical, expected pose to simplify recognition in
the following layers. Notably, spatial transformers can be trained with standard back-propagation,
allowing for end-to-end training of the models they are injected in.
The number of parameters used for spatial transformation are 6(for an affine transformation θ is 6-dimensional).

The existing code was compatible with MNIST dataset of 28x28x1. This was modified to make it compatible with CIFAR10 dataset 32x32x3. The kernel and strides were kept as is and appropriate calculation was done to derive the size of FC layer(500X1) and image size after localization (-1, 10 * 4 * 4).

The modified code can be found [here](https://colab.research.google.com/drive/1WVgU0HZhsgBolyAYCDAxY0rhlg-8ecGc)


### Part -2
The model was trained for 50 epochs.



## Result

The model was trained for 50 epochs -
Highest train accuracy achievd is 64%.
Highest test accuracy achieved is 57%.

Few sample transformed(corrected) images by the Encoder(Spatial transformation block) is as below - 


![](/Images/S12_Images/Transformed_Images.png)

