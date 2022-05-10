## 						DETR for Panoptic segmentation

Work for Panoptic Segmentation using DETR can be broken down as below -
1. Finding dataset of images
2. Annotating the images and producing binary mask
3. Using the binary mask creating coco json file with categoroes = number of classes + stuff class (isthing = 0). In our case we had the following classes
	-	rebar
	-	crack
	-	spall
	- 	stuff/background
4. Train the DETR model for BBOX. The BBOX must detect all classes + stuff class.
The colab notebook for BBOX training can be found [here](https://colab.research.google.com/drive/1bIaMU19oRXRZuNMsz-VQL12YrjaZACNC).

## For segmentation
5. Create ground-truth png and json files.
The code to generate ground-truth and corresponding train/val json can be found [here](https://colab.research.google.com/drive/1J9EeYhxhTxprXhtAxUuyz5OhZ-i2T6P2)

6. Freeze the BBOX trained DETR weights and use it to train the DETR for segmentation.
The colab notebook for Segmentation training can be found [here](https://colab.research.google.com/drive/1byQxIFpL10DVO6QrR_mDdni45BEzAjtC).

7. The BBOX and Segmentation log files can be found [here](/Capstone/Logs)

# Results
##The final images are in the following order -

##Original Image -> Ground Truth -> BBOX -> Panoptic Segmented

![](/Images/Capstone_3/capstone_final_images.png)


# Logs

## Log for BBOX trained for 175 epochs
![](/Images/Capstone_3/BBOX_Log1.png)

![](/Images/Capstone_3/BBOX_Log2.png)

![](/Images/Capstone_3/BBOX_Log3.png)


## Log for Segmentation as below

![](/Images/Capstone_3/SEG_Log1.png)

![](/Images/Capstone_3/SEG_Log2.png)

![](/Images/Capstone_3/SEG_Log3.png)


