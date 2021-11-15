# Session - 7 Assignment

## Requirement

1. To create a network architecture  C1->C2->C3->C4->0utput (Dilated kernels to be used where appropriate)
2. Total RF must be more than 52
3. Two of the layers must use Depthwise Separable Convolution
4. One of the layers must use Dilated Convolution
5. Mandatory use of GAP
6. Use of albumentation library as below:
- horizontal flip
- shiftScaleRotate
- coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, 
                     fill_value=(mean of your dataset), mask_fill_value = None)  
- grayscale

7. Target 87% accuracy with less than 100k total parameters (There is no limitation in use of number of epochs)


## Approach

The network class was modified to accept a parameter that would decide the type of normalization to be used.
Parameter values as below:
B - For BN + L1 Regularization
L - For Layer Normalization
G - For group Normalization (Group of 4 per layer has been used for this assignment)

The Network class has been saved in a different file named as model.py
A Notebook file was created to create 3 versions of the model.

For L1 regularization the Train function has been modified to use mse loss function. The predicted tensor output from the model has been reshaped(Used argmax) to match with the target tensor shape. This was needed to make it compatible with mse loss function. The train function has also been modified to return a list of train loss and train accuracy.

The Test fucntion has been modified to return a list of test loss and test accuracy. This was needed to plot the graph for all 3 models.


## Model Summary

### Version -1 (Model with BN + L1)
<pre>
