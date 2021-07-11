# Garment-Classifier
## Classifies Garment Attribute

### Dataset Structure 

```plain
├── images    <-- 1782 train data
└── attributes.csv     <-- 2238 rows and 4 columns
```
### Packages
Pytorch, Torchvision, cv2, pandas, numpy, os, time, random

Run main.py file after changing the file locations to train and test the model and save the Output.csv file.

### Description
Given the limited dataset and multiple detections, transfer learning using pretrained vgg11 network was done.
VGG11 served as the feature extraction layer followed by a detection head having 3 independent detections for neck, sleeve and pattern.

The AdaptiveAvgPool layer and the dense layer was removed. Removing AvgPool layer improved the accuracy which still remains low at 40 for the neck, 61 for the sleeve and 64 for the patten.
The low accuracy could be due to directly using transfer learning as not model is not specifically build to detect garments and may be outputing patterns which are irrelevant in out use case.

Deleting top layers in the VGG may help as we may get ore raw pattern signal over which we can apply self supervised training to make the model learn relevant patterns.

Given extreme data imbalance in every attribute, the crossentropy loss was weighted according to the weight matrix given by total number of images/number of images for the particular class.

No data augmentation was added but, a provision for gausian blur and random flip has been added. Gausian blur would help with pattern class detection.

All the nan values were made a new class and while intervening, predictions of this class were given nan values.

### Future Work
Apply self supervised pre-training to get better features and also generate model data after augmentation. Also removing the top convolutional layers in the VGG11 would help as the network can train upon these primitive patterns.

### Results 
Output.csv file has been added in the repository, and the model parameters can be downloaded from https://drive.google.com/file/d/1EA92FIFfmbdMfx-PSgACva4nNGQE6aW_/view?usp=sharing . On the training data it shows an accuracy of close to 90% and as the input for output.cvs has been taken from that, it should be the same. FOr validation set it ha the accuracy as given in the description section.

I an lso attaching the colab notebook where results are more visible.