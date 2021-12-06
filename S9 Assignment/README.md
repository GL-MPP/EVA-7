# Session - 9 Assignment

## Requirement

Create a custom ResNet architecture for CIFAR10 that has the following architecture:
1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
2. Layer1 -
- X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
- R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
- Add(X, R1)
3. Layer 2 - Conv 3x3 [256k] >> MaxPooling2D >> BN >> ReLU
4. Layer 4 -
- X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
- R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
- Add(X, R2)
- MaxPooling with Kernel Size 4
- FC Layer 
5. SoftMax
6. Uses One Cycle Policy such that:
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = To be derived
- LRMAX = To be derived
- No Annihilation
7. Uses the transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
8. Batch size = 512

## Approach

There are 3 main parts to this excercise.

Part -1 (Generating Image set with Albumentation libraries)

Part -2 (Updating the network skeleton and training)

Part -3 (Generating 10 misclassified images and its corresponding gradcam images)

### Part -1
The Albumentation library was used to generate additional images on top of training data set by using following modifications on the train dataset- 
- The dataset was first imported with only transformation being converting to tensor
- The mean and std was then derived separately for train and test dataset
- The dataset was reloaded with Normalized values.
- Using albumentation libraries RandomCrop and Cutout rules were applied. For cutout the mean calculation was appropriately done to sync with normalized data. This was needed to fill the cutout section with mean value of pixels.

The test data set was only converted to tensor and normalized using mean and std.

### Part -2
The existing Resnet network class was updated to include the following.
1. The network was segregated into 3 parts Input -> conv+Block till image 8x8 -> output layer 1x10.
2. This was needed to enable segregated backpropagation for gradcam calulation.
3. The model was trained and during each training epoch a new set of augmented imagest was genarated.

The model was trained 
Model Summary as below:
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          36,864
       BatchNorm2d-5           [-1, 64, 32, 32]             128
            Conv2d-6           [-1, 64, 32, 32]          36,864
       BatchNorm2d-7           [-1, 64, 32, 32]             128
        BasicBlock-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 64, 32, 32]          36,864
      BatchNorm2d-10           [-1, 64, 32, 32]             128
           Conv2d-11           [-1, 64, 32, 32]          36,864
      BatchNorm2d-12           [-1, 64, 32, 32]             128
       BasicBlock-13           [-1, 64, 32, 32]               0
           Conv2d-14          [-1, 128, 16, 16]          73,728
      BatchNorm2d-15          [-1, 128, 16, 16]             256
           Conv2d-16          [-1, 128, 16, 16]         147,456
      BatchNorm2d-17          [-1, 128, 16, 16]             256
           Conv2d-18          [-1, 128, 16, 16]           8,192
      BatchNorm2d-19          [-1, 128, 16, 16]             256
       BasicBlock-20          [-1, 128, 16, 16]               0
           Conv2d-21          [-1, 128, 16, 16]         147,456
      BatchNorm2d-22          [-1, 128, 16, 16]             256
           Conv2d-23          [-1, 128, 16, 16]         147,456
      BatchNorm2d-24          [-1, 128, 16, 16]             256
       BasicBlock-25          [-1, 128, 16, 16]               0
           Conv2d-26            [-1, 256, 8, 8]         294,912
      BatchNorm2d-27            [-1, 256, 8, 8]             512
           Conv2d-28            [-1, 256, 8, 8]         589,824
      BatchNorm2d-29            [-1, 256, 8, 8]             512
           Conv2d-30            [-1, 256, 8, 8]          32,768
      BatchNorm2d-31            [-1, 256, 8, 8]             512
       BasicBlock-32            [-1, 256, 8, 8]               0
           Conv2d-33            [-1, 256, 8, 8]         589,824
      BatchNorm2d-34            [-1, 256, 8, 8]             512
           Conv2d-35            [-1, 256, 8, 8]         589,824
      BatchNorm2d-36            [-1, 256, 8, 8]             512
       BasicBlock-37            [-1, 256, 8, 8]               0
           Conv2d-38            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-39            [-1, 512, 4, 4]           1,024
           Conv2d-40            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-41            [-1, 512, 4, 4]           1,024
           Conv2d-42            [-1, 512, 4, 4]         131,072
      BatchNorm2d-43            [-1, 512, 4, 4]           1,024
       BasicBlock-44            [-1, 512, 4, 4]               0
           Conv2d-45            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-46            [-1, 512, 4, 4]           1,024
           Conv2d-47            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-48            [-1, 512, 4, 4]           1,024
       BasicBlock-49            [-1, 512, 4, 4]               0
        AvgPool2d-50            [-1, 512, 1, 1]               0
           Linear-51                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.75
Params size (MB): 42.63
Estimated Total Size (MB): 54.39
----------------------------------------------------------------
</pre>

### Part -3

10 misclassified images were generated and for each misclassified image a gradcam image was generated for the misclassified class the model predicted.

![](/Images/Missclassified_img.jpg)

Gradcam images - 

![](/Images/Gradcam_img.jpg)


## Result

The model was trained for 20 epochs -
- Highest Training Accuracy achieved - 92.43%
- Highest Test Accuracy achieved - 50.23 at epoch 14.


![](/Images/Train_test_graph.png)
