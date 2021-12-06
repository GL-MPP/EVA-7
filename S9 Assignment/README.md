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
7. Uses the transform - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
8. Batch size = 512

## Approach

There are 3 main parts to this excercise.

Part -1 (Creating Image transformation rules -> RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8))

Part -2 (Creating a Custom Resnet model)

Part -3 (Finding the optimum Max LR and passing it as a parameter to One Cycle policy Scheduler to obtain maximum model accuracy)

### Part -1
The image transformation rules were derived by using a mixture of torchvison.transforms library and Albumenations.
torchvison.transforms was used for:
- Converting to Tensor
- Random crop 32x32 with padding of 4
- Horizontal flip

Albumentation libraries was used for Cutout and Normalizing the images using Mean, Std and Max. As the image pixel range changes after being converted to Tensor (New pixel values ranges from 0-1) the new image mean,std and max were passed to Albumentation library to generate Normalized images. 

The test data set was only converted to tensor and normalized using mean, std and max.

### Part -2
A custom resnet model was developed. Grouping rules were set in palce which reduced the number of parameters to 757K.
Model Summary as below:
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 3, 32, 32]              27
       BatchNorm2d-2            [-1, 3, 32, 32]               6
              ReLU-3            [-1, 3, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]             192
       BatchNorm2d-5           [-1, 64, 32, 32]             128
              ReLU-6           [-1, 64, 32, 32]               0
            Conv2d-7           [-1, 64, 32, 32]             576
       BatchNorm2d-8           [-1, 64, 32, 32]             128
              ReLU-9           [-1, 64, 32, 32]               0
           Conv2d-10          [-1, 128, 32, 32]           8,192
        MaxPool2d-11          [-1, 128, 16, 16]               0
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]           1,152
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
           Conv2d-17          [-1, 128, 16, 16]          16,384
      BatchNorm2d-18          [-1, 128, 16, 16]             256
             ReLU-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]           1,152
      BatchNorm2d-21          [-1, 128, 16, 16]             256
             ReLU-22          [-1, 128, 16, 16]               0
           Conv2d-23          [-1, 128, 16, 16]          16,384
      BatchNorm2d-24          [-1, 128, 16, 16]             256
             ReLU-25          [-1, 128, 16, 16]               0
   Residual_Block-26          [-1, 128, 16, 16]               0
           Conv2d-27          [-1, 128, 16, 16]           1,152
      BatchNorm2d-28          [-1, 128, 16, 16]             256
             ReLU-29          [-1, 128, 16, 16]               0
           Conv2d-30          [-1, 256, 16, 16]          32,768
        MaxPool2d-31            [-1, 256, 8, 8]               0
      BatchNorm2d-32            [-1, 256, 8, 8]             512
             ReLU-33            [-1, 256, 8, 8]               0
           Conv2d-34            [-1, 256, 8, 8]           2,304
      BatchNorm2d-35            [-1, 256, 8, 8]             512
             ReLU-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 8, 8]         131,072
        MaxPool2d-38            [-1, 512, 4, 4]               0
      BatchNorm2d-39            [-1, 512, 4, 4]           1,024
             ReLU-40            [-1, 512, 4, 4]               0
           Conv2d-41            [-1, 512, 4, 4]           4,608
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
             ReLU-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]         262,144
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
             ReLU-46            [-1, 512, 4, 4]               0
           Conv2d-47            [-1, 512, 4, 4]           4,608
      BatchNorm2d-48            [-1, 512, 4, 4]           1,024
             ReLU-49            [-1, 512, 4, 4]               0
           Conv2d-50            [-1, 512, 4, 4]         262,144
      BatchNorm2d-51            [-1, 512, 4, 4]           1,024
             ReLU-52            [-1, 512, 4, 4]               0
   Residual_Block-53            [-1, 512, 4, 4]               0
        MaxPool2d-54            [-1, 512, 1, 1]               0
           Linear-55                   [-1, 10]           5,130
================================================================
Total params: 757,931
Trainable params: 757,931
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.32
Params size (MB): 2.89
Estimated Total Size (MB): 14.23
----------------------------------------------------------------
</pre>

### Part -3

The model was initialized with StepLR step size = 1 and gamma = 1.069. Otimizer used was Adam.
The model was trained for one epoch with the above LR scheduler to derive the Max LR when the Loss is minimum. 
The graph gerenated LR vs Training Loss is as bleow-

![](/Images/LRVsLoss.png)

## Result

The model was trained for 20 epochs -
- Highest Training Accuracy achieved - 92.43%
- Highest Test Accuracy achieved - 50.23 at epoch 14.


![](/Images/Train_test_graph.png)
