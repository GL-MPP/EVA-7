# Session - 11 Assignment

## Requirement

OpenCV Yolo:  [SOURCE](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

- Run this above code on your laptop or Colab.
- Take an image of self, holding another object which is there in COCO data set
- Run this image through the code above. 
- Upload/display the annotated image by YOLO. 

Train Custom Dataset on Colab for YoloV3
Refer to this Colab File:  [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS#scrollTo=oJWPCDSv0gw3)

Refer to this GitHub  [Repo](https://github.com/theschoolofai/YoloV3)
Download this dataset (Links to an external site.). 

- Collect and add 25 images for the following 4 classes into the dataset shared:
class names are in custom.names file. 

- Steps are explained in the README.md file on github repo link should be followed.

- Once additional 100 images are added, train the model

## Approach

There are 5 parts to this exercise.

Part -1 Preparing and manually annotating images to prepare corresponding image labels.

Part -2 Updating config file and placing the annotated images and labels in appropriate folders

Part -3 Training the model

Part -4 Displaying the newly added images annoated images by Yolo.

Part -5 Downlaoding a short video, breaking the video into frames of images. Annotating the images through yolo and rebuilding the video using annotated images.

### Part -1
Images were downloaded 25 images per class(for 4 classes). The images were then annotated manually using the annotaion tool. The images and generated labels were placed in appropriate folders.

### Part -2
The config file was updated to make number of classes = 4. There were extra image information in train.txt which had to be removed to make the structure ready for training.

### Part -3
The model was trained on existing and newly added annotated images.

### Part -4
The yolo annotated images were generated displayed below.

### Part -5
A short video was downloaded from youtube. The video was broken into image fragments which was fed to yolo for annotation. The video was reconstructed using yolo annotated images.


## Result

The model was trained for 20 epochs -
The yolo annotated images generated are as below -


![](/Images/S11 Yolo/Boots1.jpg)

![](/Images/S11 Yolo/Boots2.jpg)

![](/Images/S11 Yolo/Boots3.jpg)

![](/Images/S11 Yolo/Boots4.jpg)

![](/Images/S11 Yolo/HHat1.jpg)

![](/Images/S11 Yolo/HHat2.jpg)

![](/Images/S11 Yolo/HHat3.jpg)

![](/Images/S11 Yolo/HHat4.jpg)

![](/Images/S11 Yolo/Mask1.jpg)

![](/Images/S11 Yolo/Mask2.jpg)

![](/Images/S11 Yolo/Mask3.jpg)

![](/Images/S11 Yolo/Mask4.jpg)

![](/Images/S11 Yolo/Vest1.jpg)

![](/Images/S11 Yolo/Vest2.jpg)

![](/Images/S11 Yolo/Vest3.jpg)

![](/Images/S11 Yolo/Vest4.jpg)
