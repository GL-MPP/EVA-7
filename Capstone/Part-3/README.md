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

## Results
### The model output images are in the following order -

### Original Image -> Ground Truth -> BBOX -> Panoptic Segmented

![](/Images/Capstone_3/capstone_final_images1.png)

![](/Images/Capstone_3/capstone_final_images2.png)

![](/Images/Capstone_3/capstone_final_images3.png)

![](/Images/Capstone_3/capstone_final_images4.png)
![](/Images/Capstone_3/capstone_final_images5.png)
![](/Images/Capstone_3/capstone_final_images6.png)
![](/Images/Capstone_3/capstone_final_images7.png)
![](/Images/Capstone_3/capstone_final_images8.png)
![](/Images/Capstone_3/capstone_final_images9.png)
![](/Images/Capstone_3/capstone_final_images10.png)
![](/Images/Capstone_3/capstone_final_images11.png)
![](/Images/Capstone_3/capstone_final_images12.png)
![](/Images/Capstone_3/capstone_final_images13.png)
![](/Images/Capstone_3/capstone_final_images14.png)
![](/Images/Capstone_3/capstone_final_images15.png)
![](/Images/Capstone_3/capstone_final_images16.png)
![](/Images/Capstone_3/capstone_final_images17.png)
![](/Images/Capstone_3/capstone_final_images18.png)
![](/Images/Capstone_3/capstone_final_images19.png)
![](/Images/Capstone_3/capstone_final_images20.png)
![](/Images/Capstone_3/capstone_final_images21.png)
![](/Images/Capstone_3/capstone_final_images22.png)
![](/Images/Capstone_3/capstone_final_images23.png)
![](/Images/Capstone_3/capstone_final_images24.png)
![](/Images/Capstone_3/capstone_final_images25.png)
![](/Images/Capstone_3/capstone_final_images26.png)
![](/Images/Capstone_3/capstone_final_images27.png)
![](/Images/Capstone_3/capstone_final_images28.png)
![](/Images/Capstone_3/capstone_final_images29.png)
![](/Images/Capstone_3/capstone_final_images30.png)
![](/Images/Capstone_3/capstone_final_images31.png)
![](/Images/Capstone_3/capstone_final_images32.png)
![](/Images/Capstone_3/capstone_final_images33.png)
![](/Images/Capstone_3/capstone_final_images34.png)
![](/Images/Capstone_3/capstone_final_images35.png)
![](/Images/Capstone_3/capstone_final_images36.png)
![](/Images/Capstone_3/capstone_final_images37.png)
![](/Images/Capstone_3/capstone_final_images38.png)
![](/Images/Capstone_3/capstone_final_images39.png)
![](/Images/Capstone_3/capstone_final_images40.png)
![](/Images/Capstone_3/capstone_final_images41.png)
![](/Images/Capstone_3/capstone_final_images42.png)
![](/Images/Capstone_3/capstone_final_images43.png)
![](/Images/Capstone_3/capstone_final_images44.png)
![](/Images/Capstone_3/capstone_final_images45.png)
![](/Images/Capstone_3/capstone_final_images46.png)
![](/Images/Capstone_3/capstone_final_images47.png)
![](/Images/Capstone_3/capstone_final_images48.png)
![](/Images/Capstone_3/capstone_final_images49.png)
![](/Images/Capstone_3/capstone_final_images50.png)
![](/Images/Capstone_3/capstone_final_images51.png)
![](/Images/Capstone_3/capstone_final_images52.png)
![](/Images/Capstone_3/capstone_final_images53.png)
![](/Images/Capstone_3/capstone_final_images54.png)
![](/Images/Capstone_3/capstone_final_images55.png)
![](/Images/Capstone_3/capstone_final_images56.png)
![](/Images/Capstone_3/capstone_final_images57.png)
![](/Images/Capstone_3/capstone_final_images58.png)
![](/Images/Capstone_3/capstone_final_images59.png)
![](/Images/Capstone_3/capstone_final_images60.png)
![](/Images/Capstone_3/capstone_final_images61.png)
![](/Images/Capstone_3/capstone_final_images62.png)
![](/Images/Capstone_3/capstone_final_images63.png)
![](/Images/Capstone_3/capstone_final_images64.png)
![](/Images/Capstone_3/capstone_final_images65.png)
![](/Images/Capstone_3/capstone_final_images66.png)
![](/Images/Capstone_3/capstone_final_images67.png)
![](/Images/Capstone_3/capstone_final_images68.png)
![](/Images/Capstone_3/capstone_final_images69.png)
![](/Images/Capstone_3/capstone_final_images70.png)



## Logs

### Log for BBOX trained for 175 epochs
![](/Images/Capstone_3/BBOX_Log1.png)

![](/Images/Capstone_3/BBOX_Log2.png)

![](/Images/Capstone_3/BBOX_Log3.png)


### Log for Segmentation as below

![](/Images/Capstone_3/SEG_Log1.png)

![](/Images/Capstone_3/SEG_Log2.png)

![](/Images/Capstone_3/SEG_Log3.png)


