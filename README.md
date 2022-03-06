# xray
A program to detect pneumonia in lungs using a convolutional neural network.
# Network
The network consists of 5 convolutional layers and 3 linear layers. It will output a 1 or a 0, 1 if it detects pneumonia, 0 if it doesn't 
# Testing
On the validation set i got a score between 87.5% and 100%. The problem with the validation set is that it consists of only 16 images (very small sample size). Therefore it would make sense to include some of the training data in the validation set to get a more accurate score, and then of course removing that data from the training set. # Data
All the data is taken from kaggle. The images is first resized to 250x200 pixels, grayscaled and then saved into numpy arrays to be saved in npy files. I did this so I wouldn't have to do it again and thus saved time.     
