# Session - 8 Assignment

## Requirement

Create a structured repository as below -
1. Models folder - This folder will contain all future models. 
2. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
3. Create a main.py file to contain the following funtionalities
- training and test loops
- data split between test and train
- epochs
- batch size
- optimizer initialization

4. Create utils.py file and add utilities like:
- image transforms
- gradcam gereation rules
- misclassification code
- tensorboard related stuff

5. The colab notebook will only call these files/utilities and nothing is coded in the colab notebook.
6. Train resnet18 for 20 epochs on the CIFAR10 dataset
7. To apply following tranformations while training - 
- RandomCrop(32, padding=4)
- CutOut(16x16)
8. To show loss curves for test and train datasets
9. To show a gallery of 10 misclassified images
10. To show gradcam output on 10 misclassified images. Apply GradCAM on a channel that is more than 7x7.

## Approach

There are 3 main parts to this excercise.

Part -1 (Generating Image set with Albumentation libraries)

Part -2 (Developing and updating the network skeleton)

### Part -1
The Albumantation library was used to generate additinal images on top of training data set by using following modofications to existing images- 
- Horizontal Flip
- ShiftScale Rotate
- Coarse Dropout
- Grayscale

Because we needed to Grayscale the use of existing coloured dataset was not needed. Hence, at first the whole training dataset was converted to grayscale and then these gray-scaled images were modified using - Horizontal Flip, ShiftScale Rotate and Coarse Dropout.
Thus, for each image (Gray scaled image) we had 3 additional versions. However, there being 50k training images adding 3 images to every image resulted in 200k training images. Because of restrictions in hardware this was selectively done i.e. few images were horizontally flipped, few were shift rotated and the rest were modified using coarse Drop out.
The total training image set had 50K gray-scaled images +~ 30K modified/augmented images Total being around 80K images. The training images were then normalized.

On similar lines the test images were gray-scaled and normalized.

### Part -2
The network class was created to include the following.
1. Depthwise separable convolution layer.
2. Dialated or Altrous convolution.
3. Normal convloution
4. GAP

The model was trained 
Model Summary as below:
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 3, 32, 32]              27
       BatchNorm2d-2            [-1, 3, 32, 32]               6
              ReLU-3            [-1, 3, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]              96
       BatchNorm2d-5           [-1, 32, 32, 32]              64
           Dropout-6           [-1, 32, 32, 32]               0
              ReLU-7           [-1, 32, 32, 32]               0
            Conv2d-8           [-1, 32, 32, 32]             288
       BatchNorm2d-9           [-1, 32, 32, 32]              64
             ReLU-10           [-1, 32, 32, 32]               0
           Conv2d-11           [-1, 64, 32, 32]           2,048
      BatchNorm2d-12           [-1, 64, 32, 32]             128
             ReLU-13           [-1, 64, 32, 32]               0
  ConvTranspose2d-14          [-1, 128, 40, 40]          73,728
      BatchNorm2d-15          [-1, 128, 40, 40]             256
             ReLU-16          [-1, 128, 40, 40]               0
           Conv2d-17           [-1, 10, 32, 32]          11,520
      BatchNorm2d-18           [-1, 10, 32, 32]              20
             ReLU-19           [-1, 10, 32, 32]               0
        AvgPool2d-20             [-1, 10, 1, 1]               0
================================================================
Total params: 88,245
Trainable params: 88,245
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.24
Params size (MB): 0.34
Estimated Total Size (MB): 8.59
----------------------------------------------------------------
</pre>


## Result

Because of time contraints the number of epochs were restricted to only 8. The overall accuracy was 67% and accuracy of predicted classes is as below - 

Accuracy of plane : 77 %

Accuracy of   car : 82 %

Accuracy of  bird : 51 %

Accuracy of   cat : 55 %

Accuracy of  deer : 67 %

Accuracy of   dog : 72 %

Accuracy of  frog : 70 %

Accuracy of horse : 75 %

Accuracy of  ship : 60 %

Accuracy of truck : 66 %


## Inference

Given the improvement in accuracy for only 8 epochs, if the model is trained further the accuracy should improve.
