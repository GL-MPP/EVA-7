# Session - 10 Assignment -B

## Requirement

Download  COCO object detection dataset's schema text file.

Identify the values present in the dataset.

Display class distribution with a graph.

Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.


## Analysis

The coco data set has set of image information like image size (Height and Width), labels(Class - ID) and bonding box(Diagonal co-ordinates). 

There are 79 classes and the class distribution graph is as below -

![](/Images/Kmeans_class_distribution.png)

The data set was loaded as a dataframe in a notebook and the bonding box size (Height and Width) was calculated using the diagonal coordinates. The bonding box was then plotted with one corner being (0,0) as a scattered plot.

![](/Images/BBOX.png)

KMeans clustering algorythm was run on X,Y coordiantes to find the optimum number of clusters

![](/Images/ELBOW.png)

Then, for cluster values = 3,4,5,6 the centroids were calculated and anchor boxes were generated.

![](/Images/Kmeans_3.png)

![](/Images/Kmeans_4.png)

![](/Images/Kmeans_5.png)

![](/Images/Kmeans_6.png)
