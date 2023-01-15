# Dataset-BCI-competition-iv-set1
This code is written in Python, and it uses several libraries, including pandas, numpy, os, tensorflow (with the Keras API), cv2, and matplotlib. These libraries are imported at the beginning of the code.

The main purpose of this code is to perform experiments on Generative Adversarial Networks (GANs) for Brain-Computer Interface (BCI) systems. It loads the spectral images from a directory that contain different subjects' spectral images, then it creates a dataset from these images by resize these images and normalizing them between 0 and 1. The class_name list is also created to hold the name of the class corresponding to each image.

The code uses KFold method from sklearn library to split the data into k folds where k=5 in this code (the number of subjects) to be used in the training and validation stages, also it defines a function to create the dataset which takes the folder directory as an input argument.

The code then uses the Keras API, a high-level neural networks API in TensorFlow, to define, train, and evaluate CNN models, GAN models, and GAN-CNN models. It uses several of the standard layers in Keras, such as Dense, Dropout, Flatten, Conv2D, and MaxPool2D. The code also uses several other functions and classes from the Keras and TensorFlow libraries to compile and train the models, such as Model, Sequential, SGD, MaxNorm, EarlyStopping, and ModelCheckpoint.

Additionally, it uses os.path, sys and numpy.random libraries to create results folder and fix random seed for reproducibility. Finally, it uses os.path and os.makedirs functions to create folder for results, Also it uses os.path.exists() to check if the folder already exists or not.

The results are the classification accuracy for each model, the models are trained and tested on different subjects and accuracy for each subject is saved in two lists (CNN_acc, GAN_acc) to compare between GAN-CNN models and CNN models.
