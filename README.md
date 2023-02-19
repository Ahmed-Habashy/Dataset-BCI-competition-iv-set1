# Title Image Classification using Convolutional Neural Network and Generative Adversarial Network
Description:
This code is for classifying spectrogram images of Motor Movement/Imagery tasks using a Convolutional Neural Network (CNN) and Generative Adversarial Network (GAN). The code loads the dataset of spectrogram images of motor tasks, preprocesses the images, trains the CNN model using k-fold cross-validation and predicts the class labels of the test set. The GAN is used to generate images of the two classes that are used in the binary classification problem of the CNN model.

# Dependencies
numpy
os
tensorflow
opencv-python
matplotlib
keras
sklearn
PIL
Dataset:
The dataset used for this code is the BCI-IV 1 dataset, which contains the EEG signals of 9 subjects performing Motor Movement/Imagery tasks. The dataset is preprocessed and transformed into spectrogram images using the Short Time Fourier Transform (STFT).

# Usage
The main file is the BCI_IV_1_GAN_CNN.py. Run the file to train and test the CNN and GAN models.
The code can be modified to use other datasets of spectrogram images.
The code can be modified to change the parameters of the CNN and GAN models.

# Credits
The code is adapted from the following sources:
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/
