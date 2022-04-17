# xray
A program to detect pneumonia in lungs using a convolutional neural network.
# Network
The network consists of 4 convolutional layers and 3 linear layers. I decided to use dropout on the linear layers in order to avoid overfitting the model.
# Testing
The problem with the validation set is that it consists of only 16 images (very small sample size). Therefore it would make sense to include some of the training data in the validation set to get a more accurate score, and then of course removing that data from the training set. I increased the validation set to a sample size of 100 images and ended up with an accuracy of **98%**.

# Data  
All the data is taken from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The images is resized to 150x150 pixels and then saved into numpy arrays. All of the data processing code is in the **data.py** file.     
