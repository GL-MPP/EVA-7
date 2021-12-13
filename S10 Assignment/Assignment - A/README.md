# Session - 10 Assignment -A

## Requirement

Download this  TINY IMAGENET dataset. 
Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 

## Approach

There are 3 main parts to this exercise.

Part -1 Preparing dataset

Part -2 Updating Resnet18 model to fit the new dataset and classes

Part -3 (Finding the optimum Max LR and passing it as a parameter to One Cycle policy Scheduler to obtain maximum model accuracy)

### Part -1
For ImageFolder to work, images in training and validation folders must be arranged in the following structure:
==> train/cat/img1.png

==> train/cat/img2.png

==> train/cat/img3.png

==> train/cat/img4.png

==> val/cat/img_1.png

==> val/cat/img_2.png

==> val/cat/img_3.png


It is found that the training folder meets the structure needed for ImageLoader but the validation folder does not. The images in the validation folder are all saved within a single folder, so we need to reorganize them into sub-folders based on their labels.
The validation folder contains a val_annotations.txt file which comprises six tab-separated columns: filename, class label, and details of the bounding box (x,y coordinates, height, width).

![](/Images/TinyImagevaldef.png)

We extract the first two columns so that we can save the pairs of filename and corresponding class labels in a dictionary. After that, we carry out the folder path reorganization.

The images were normalized to mean values of (0.485, 0.456, 0.406) and standard deviation values of (0.229, 0.224, 0.225).
On top of that, transformation rules were added (e.g. center crops, random flips, etc.) to augment the image dataset and improve model performance.

### Part -2
Resnet18 model was updated to add number of classes to 200.
Model Summary as below:
<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,728
       BatchNorm2d-2         [-1, 64, 224, 224]             128
            Conv2d-3         [-1, 64, 224, 224]          36,864
       BatchNorm2d-4         [-1, 64, 224, 224]             128
            Conv2d-5         [-1, 64, 224, 224]          36,864
       BatchNorm2d-6         [-1, 64, 224, 224]             128
        BasicBlock-7         [-1, 64, 224, 224]               0
            Conv2d-8         [-1, 64, 224, 224]          36,864
       BatchNorm2d-9         [-1, 64, 224, 224]             128
           Conv2d-10         [-1, 64, 224, 224]          36,864
      BatchNorm2d-11         [-1, 64, 224, 224]             128
       BasicBlock-12         [-1, 64, 224, 224]               0
           Conv2d-13        [-1, 128, 112, 112]          73,728
      BatchNorm2d-14        [-1, 128, 112, 112]             256
           Conv2d-15        [-1, 128, 112, 112]         147,456
      BatchNorm2d-16        [-1, 128, 112, 112]             256
           Conv2d-17        [-1, 128, 112, 112]           8,192
      BatchNorm2d-18        [-1, 128, 112, 112]             256
       BasicBlock-19        [-1, 128, 112, 112]               0
           Conv2d-20        [-1, 128, 112, 112]         147,456
      BatchNorm2d-21        [-1, 128, 112, 112]             256
           Conv2d-22        [-1, 128, 112, 112]         147,456
      BatchNorm2d-23        [-1, 128, 112, 112]             256
       BasicBlock-24        [-1, 128, 112, 112]               0
           Conv2d-25          [-1, 256, 56, 56]         294,912
      BatchNorm2d-26          [-1, 256, 56, 56]             512
           Conv2d-27          [-1, 256, 56, 56]         589,824
      BatchNorm2d-28          [-1, 256, 56, 56]             512
           Conv2d-29          [-1, 256, 56, 56]          32,768
      BatchNorm2d-30          [-1, 256, 56, 56]             512
       BasicBlock-31          [-1, 256, 56, 56]               0
           Conv2d-32          [-1, 256, 56, 56]         589,824
      BatchNorm2d-33          [-1, 256, 56, 56]             512
           Conv2d-34          [-1, 256, 56, 56]         589,824
      BatchNorm2d-35          [-1, 256, 56, 56]             512
       BasicBlock-36          [-1, 256, 56, 56]               0
           Conv2d-37          [-1, 512, 28, 28]       1,179,648
      BatchNorm2d-38          [-1, 512, 28, 28]           1,024
           Conv2d-39          [-1, 512, 28, 28]       2,359,296
      BatchNorm2d-40          [-1, 512, 28, 28]           1,024
           Conv2d-41          [-1, 512, 28, 28]         131,072
      BatchNorm2d-42          [-1, 512, 28, 28]           1,024
       BasicBlock-43          [-1, 512, 28, 28]               0
           Conv2d-44          [-1, 512, 28, 28]       2,359,296
      BatchNorm2d-45          [-1, 512, 28, 28]           1,024
           Conv2d-46          [-1, 512, 28, 28]       2,359,296
      BatchNorm2d-47          [-1, 512, 28, 28]           1,024
       BasicBlock-48          [-1, 512, 28, 28]               0
           Linear-49                  [-1, 200]         102,600
================================================================
Total params: 11,271,432
Trainable params: 11,271,432
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 551.25
Params size (MB): 43.00
Estimated Total Size (MB): 594.82
----------------------------------------------------------------
</pre>

### Part -3

The model was run in two modes:
1. Train mode = N
2. Train mode = Y

- In Train mode = N the model was initialized with StepLR step size = 1 and gamma = 1.069. Optimizer used was Adam.
The model was trained for one epoch with the above LR scheduler to derive the Max LR when the Loss is minimum. 
The graph gerenated LR vs Training Loss is as bleow-

![](/Images/S10LRVsLoss.png)

- In Train Mode = Y
Using several trial Train Runs the optimum Max LR was found to be 0.00203. The model was then trained using OneCycleLR scheduler for each epoch
with Max LR being at 4th epoch without annihilation. The training was limitied to 11 epochs because of time limitations.

## Result

The model was trained for 11 epochs -
- Highest Training Accuracy achieved - 57.25 at 9th epoch
- Highest Test Accuracy achieved - 49.22 at 11th epoch.


![](/Images/S10_Train_Test_Acc.png)
