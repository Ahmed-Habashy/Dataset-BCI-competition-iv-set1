
import numpy as np
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D, MaxPooling2D, Input, InputLayer,Reshape, Conv2DTranspose
from keras.utils import to_categorical 
from keras import backend 
from keras.optimizers import SGD
from keras.constraints import MaxNorm
from sklearn.model_selection import  StratifiedKFold
from numpy import std
from numpy import mean
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.utils import plot_model
import sys
from os import makedirs

# from numpy import expand_dims
from numpy import zeros
from numpy import ones
from keras.optimizers import Adam, RMSprop
from keras.layers import LeakyReLU, BatchNormalization, ReLU
# from keras.utils.vis_utils import plot_model
from numpy.random import randn, randint
from matplotlib import pyplot
import matplotlib.image as mpimg
from PIL import Image

# ============================= Loading Spectrum Images ========================
# defining the input images size    
IMG_WIDTH= 32
IMG_HEIGHT= 32
n_epochs = 300
GAN_epochs = 300
n_batch = 20
cnn_batch_size = 9 
cnn_epochs = 600
Ad_times = 1
R_nfolds = [ 9 ]
subject = "sub_c"



    # fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)
np.random.seed(seed)


img_folder =r'gray\{}'.format(subject)   

def create_dataset(img_folder):       
    img_data_array=[]
    class_name=[]       
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=mpimg.imread(image_path)
            # image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image = (image - 127.5) / 127.5
            # image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    # extract the image array and class name
    (img_data, class_name) = (img_data_array,class_name)
    # Create a dictionary for all unique values for the classes
    target_dict={s: v for v, s in enumerate(np.unique(class_name))}
    target_dict
    # Convert the class_names to their respective numeric value based on the dictionary
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
    x=tf.cast(np.array(img_data), tf.float64).numpy()
    y=tf.cast(list(map(int,target_val)),tf.int32).numpy()
    return x, y

xx,yy = create_dataset(img_folder)
          
def fold_split(xdata,ydata,folds=10):         
    trainX = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) , IMG_HEIGHT, IMG_WIDTH))
    trainY = np.empty((int(folds),int (len(xdata)-(len(xdata)/folds)) ))
    testX = np.empty((int(folds),int (len(xdata)/folds) , IMG_HEIGHT, IMG_WIDTH))
    testY = np.empty((int(folds),int (len(xdata)/folds) ))
    sub_fold = StratifiedKFold(folds, shuffle=True, random_state=2) 
    i=0
    # ## enumerate splits
    for train, cv in sub_fold.split(xdata,ydata):
        # select data for train and test
        trainX[i,:,:,:], trainY[i,:], testX[i,:,:,:], testY[i,:] = xdata[train], ydata[train], xdata[cv], ydata[cv]
        i+=1
    return trainX, trainY, testX, testY

train_X, train_Y, testX, testY = fold_split(xx,yy)   

print ('Raw data = ',train_X.shape, train_Y.shape)
print ('Test data = ',testX.shape, testY.shape)
 
  
#%% ================================== GAN ===========================
def class_imgs(x,y):
    class_count = 0
    for i in range( len(y[nfolds]) ):
            if (y[nfolds,i] == 1 ):
                class_count += 1
    cl1 = x[nfolds,0:class_count, :,:]
    cl2 = x[nfolds,class_count:, :,:]
    return cl1,cl2

# define the standalone discriminator model
def define_discriminator(in_shape=(IMG_HEIGHT, IMG_WIDTH,1)):
    model = Sequential()
    # normal
    model.add(Conv2D(128, 3, strides=(2,2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, 3, strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for  image
    n_nodes = 64 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 64)))
   
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # output layer
    model.add(Conv2D(1, 4, activation='tanh', padding='same'))
    return model
# =================================================
# define discriminator model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='gray/{0} GAN_results/fold_{}/GAN_discriminator_plot.png'.format(subject), show_shapes=True, show_layer_names=True)
# define the generator model
model = define_generator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='gray/{0} GAN_results/fold_{}/GAN_generator_plot.png'.format(subject), show_shapes=True, show_layer_names=True)

#%% define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential(name="GAN")
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1 = 0.5) #, beta_2 = 0.8
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# load and prepare training images
def load_real_samples(cl):
    if cl==1:
        X = x_img_cl1
    else:
        X = x_img_cl2
        cl=2            
    return X,cl

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=4):
    # scale from [-1,1] to [0,1]
    # examples = (examples + 1) / 2.0
    # plot images
    plt.figure(figsize=(IMG_HEIGHT,IMG_WIDTH))
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    pyplot.savefig('gray/{0} GAN_results/fold_{1}/plots {0}/test_GAN_batch{2}_cl{3}_{4}.png' .format(subject,nfolds, n_batch, cl, epoch+1))
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    g_model.save('gray/{0} GAN_results/fold_{1}/models {0}/test_GAN_batch{2}_cl{3}_{4}.h5'.format(subject,nfolds, n_batch, cl, epoch+1))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d1_hist, label='d-real')
    pyplot.plot(d2_hist, label='d-fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.plot(a1_hist, label='acc-real')
    pyplot.plot(a2_hist, label='acc-fake')
    pyplot.legend()
    # save plot to file
    pyplot.grid()
    pyplot.savefig('gray/{0} GAN_results/fold_{1}/test_plot_line_GAN {2}_cl{3}_loss.png' .format(subject,nfolds, n_batch, cl))
    pyplot.close()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs= n_epochs, n_batch= n_batch):
    # calculate the number of batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the total iterations based on batch and epoch
    n_steps = bat_per_epo * n_epochs
    # calculate the number of samples in half a batch
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    d_r_hist, d_f_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
        # record history
        d_r_hist.append(d_loss1)
        d_f_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        # evaluate the model performance, everry batch
        if (i+1) % 100 == 0 and (i+1) >= 200:
            summarize_performance(i, g_model, latent_dim)
    plot_history(d_r_hist, d_f_hist, g_hist, a1_hist, a2_hist)


#%% ============================= Genrator =====================================    
# create GAN  images    
def create_GAN_plot(examples):
    # plot images
    plt.figure(figsize=(IMG_HEIGHT,IMG_WIDTH))
    for i in range(len(examples)):
        makedirs('gray/{0} GAN_results/fold_{1}/GAN_dataset{3}/CL{2}'.format(subject,nfolds , cl, GAN_epochs ), exist_ok=True)
        # define subplot
        plt.axis('off')
        pyplot.gray()
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray')  # cmap = 'jet'                
        plt.savefig('gray/{0} GAN_results/fold_{1}/GAN_dataset{4}/CL{2}/cl{2}_t{3}.jpg'.format(subject,nfolds , cl , i, GAN_epochs), bbox_inches= 'tight', pad_inches= 0)
        Image.open('gray/{0} GAN_results/fold_{1}/GAN_dataset{4}/CL{2}/cl{2}_t{3}.jpg'.format(subject,nfolds , cl , i, GAN_epochs)).convert('L').save('gray/{0} GAN_results/fold_{1}/GAN_dataset{4}/CL{2}/cl{2}_t{3}.jpg'.format(subject,nfolds , cl , i, GAN_epochs))
        plt.close()

Drp1= 2
Drp2= 2
Drp3= 4
Drp4= 4
  
def create_cnn(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(8, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2
    model.add(Conv2D(8, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))                                             # dropout 3
    model.add(Flatten())
    # model.add(Dropout(0.2)) 
    model.add(Dense(100, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp4/10))                                             # dropout 4
    model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    opt = SGD(learning_rate=0.0001, momentum=0.99)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model
# model = create_cnn()
# model.summary()

def create_cnn2(): #dropout_rate1=0.0, dropout_rate2=0.0 , momentum=0, weight_constraint=0, 
    model= Sequential()
    model.add(Dropout(Drp1/10, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)) )     # dropout 1
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation= 'relu' 
                      , kernel_initializer='he_uniform' , kernel_constraint=MaxNorm(3)  ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 2
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   )) # kernel_regularizer=l2(0.001),
    model.add(Conv2D(32, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp2/10))                                             # dropout 3
    model.add(Conv2D(64, (3,3), padding='same' ,activation= 'relu' 
                      , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3)   ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(Drp3/10))           
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' , kernel_initializer='he_uniform', kernel_constraint=MaxNorm(3) ) )  
    model.add(Dropout(Drp3/10))                                             # dropout 4
    model.add(Dense(2, activation= 'softmax' , kernel_initializer='he_uniform' )) 
    opt = SGD(learning_rate=0.0001, momentum=0.99)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

# Create new CNN model
def create_cnn3():
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(IMG_HEIGHT,IMG_WIDTH, 1)))
    model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(), metrics=['accuracy'])
    return model

# #GAN-CNN Model training:
def model_training(x_data, y_data ,x_test, y_test,save_dir, sel_mod ,fig_title):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print ("Training Data= ",len(x_data) ) 
    print ("Validation Data= ",len(x_test) )
    model = sel_mod
    # 1-Times generated data:
    mcg = ModelCheckpoint(save_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model.fit(x_data, y_data, epochs=cnn_epochs , batch_size=cnn_batch_size, verbose=0 
                        ,validation_data=(x_test, y_test), callbacks=[mcg] )


    pyplot.figure()
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title(fig_title)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='lower right')
    pyplot.grid()
    pyplot.show()
    pyplot.close() 
    
    
for nfolds in range(0,1):
# for nfolds in R_nfolds:

    tf.random.set_seed(seed)
    np.random.seed(seed)
        
    x_img_cl1,x_img_cl2 = class_imgs(train_X, train_Y)
    x_test_img_cl1, x_test_img_cl2 = class_imgs(testX, testY)
    #********** GAN Batch number *********
    
    (x_tr,y_tr) = (train_X[nfolds] , train_Y[nfolds])
    (x_ev, y_ev) = (testX[nfolds], testY[nfolds])
    print ('Fold: ' , nfolds )
    print ("fold data shape = ",x_tr.shape, y_tr.shape )
    print ("Test fold data shape = ",x_ev.shape, y_ev.shape )
    
    makedirs('gray/{0} GAN_results/fold_{1}/plots {0}'.format(subject,nfolds), exist_ok=True) 
    makedirs('gray/{0} GAN_results/fold_{1}/models {0}'.format(subject,nfolds), exist_ok=True) 
    # load image data
    dataset,cl = load_real_samples(1)
    print(dataset.shape)
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)    
    # train 1st class:
    train(g_model, d_model, gan_model, dataset, latent_dim)
    
    # load image data
    dataset,cl = load_real_samples(2)
    # train 2nd class:
    train(g_model, d_model, gan_model, dataset, latent_dim)
    
    cl=1
    epochs= n_epochs         
    batch = n_batch       
    # load model
    model1 = load_model('gray/{0} GAN_results/fold_{1}/models {0}/test_GAN_batch{2}_cl{3}_{4}.h5'.format(subject,nfolds, batch, cl, epochs))
    # generate images
    latent_points1 = generate_latent_points(100,len(x_img_cl1) * Ad_times )   
    # generate images
    X_gan_lat1 = model1.predict(latent_points1)
    # plot the result
    create_GAN_plot(X_gan_lat1)
    
    cl=2
    epochs= n_epochs         
    batch = n_batch
    model2 = load_model('gray/{0} GAN_results/fold_{1}/models {0}/test_GAN_batch{2}_cl{3}_{4}.h5'.format(subject, nfolds, batch, cl, epochs))
    # generate images
    latent_points2 = generate_latent_points(100, len(x_img_cl2) *  Ad_times)
    # generate images
    X_gan_lat2 = model2.predict(latent_points2)
    # plot the result
    create_GAN_plot(X_gan_lat2)
    
    #%% ============================= GAN DATA Loading ============================================
    GAN_data= r'gray/{} GAN_results/fold_{}/GAN_dataset{}'.format(subject, nfolds, GAN_epochs)  
    
    # ======================= GAN DATA ============================    
    x_gan, y_gan =create_dataset(GAN_data)
    
    RD_AD = len(x_tr)  + Ad_times* len(x_tr) 
    x_train = np.empty((int (RD_AD) , IMG_HEIGHT, IMG_WIDTH,1))
    y_train = np.empty((int (RD_AD) )) 
    print ('GAN data shape:', x_train.shape, x_gan.shape )
    x_train = np.concatenate((x_tr, x_gan[0: Ad_times*(len(x_tr)//2)], x_gan[len(x_gan)//2: len(x_gan)//2 + Ad_times*( len(x_tr)//2) ] ))   
    y_train = np.concatenate((y_tr, y_gan[0: Ad_times*(len(y_tr)//2)], y_gan[len(y_gan)//2: len(y_gan)//2 + Ad_times*( len(y_tr)//2) ] ))  
    cnn_epochs = 600
    model_training(x_train, y_train, x_ev, y_ev, 'gray/CNN_GAN/{0}_GAN{3}_{1}_CNN3_{2}.h5'.format( subject, Ad_times, nfolds, GAN_epochs), create_cnn3(), '{}_GAN_CNN3 Model accuracy fold {} --> AD={} \n'.format(  subject, nfolds, Ad_times))        
    model_training(x_train, y_train, x_ev, y_ev, 'gray/CNN_GAN/{0}_GAN{3}_{1}_CNN3_{2}.h5'.format( subject, Ad_times, nfolds, GAN_epochs), create_cnn3(), '{}_GAN_CNN3 Model accuracy fold {} --> AD={} \n'.format(  subject, nfolds, Ad_times))        
    cnn_epochs = 100
    model_training(x_train, y_train, x_ev, y_ev, 'gray/CNN_GAN/{0}_GAN{3}_{1}_CNN3_{2}.h5'.format( subject, Ad_times, nfolds, GAN_epochs), create_cnn3(), '{}_GAN_CNN3 Model accuracy fold {} --> AD={} \n'.format(  subject, nfolds, Ad_times))        

#%%======================================= 10 fold test ==========================

def CNN_GAN_test(cnn):
    scores = list()
    for f in range(0,10):
        if cnn == 22 :
        # load the saved model
            model = load_model('gray/CNN_GAN//{0}_GAN_{1}_CNN2_model_f{2}.h5'.format( subject, Ad_times, f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('GAN2 Test Accuracy',test_acc)
            scores.append(test_acc)
        elif cnn == 33:
            model = load_model('gray/CNN_GAN/{0}_GAN{3}_{1}_CNN3_{2}.h5'.format( subject, Ad_times, f, GAN_epochs))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('GAN/CNN3 Test ',f,'=',test_acc)
            scores.append(test_acc)  
        elif cnn == 1:
            model = load_model('D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_1\MI GAN\GAN_classes\cDCGAN\CNN_cDCGAN\{0}_CNN_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)
        elif cnn == 2:
            model = load_model('D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_1\MI GAN\GAN_classes\cDCGAN\CNN_cDCGAN\{0}_CNN2_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN2: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)
        elif cnn == 3:
            model = load_model('D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_1\MI GAN\GAN_classes\cDCGAN\CNN_cDCGAN\{0}_CNN3_{1}.h5'.format( subject,f))
            test_loss, test_acc= model.evaluate(testX[f],testY[f],verbose=0)
            print('CNN3: ',f,'  Accuracy',test_acc)
            scores.append(test_acc)
   
    print('\n >>>> {0} Accuracy: mean={1} std={2}, n={3}' .format (subject, mean(scores)*100, std(scores)*100, len(scores)))
    print ('*************************************')
    ## box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    plot_model(model, show_shapes=True, expand_nested=True)
    return scores

sub_cnn1_acc = CNN_GAN_test(cnn=1)
sub_cnn2_acc = CNN_GAN_test(cnn=2)
sub_gan2_acc = CNN_GAN_test(cnn=22)
sub_cnn3_acc = CNN_GAN_test(cnn=3)
sub_gan3_acc = CNN_GAN_test(cnn=33)
