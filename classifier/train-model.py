#!/usr/bin/python

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math

## Import the keras API
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

from sklearn.model_selection import train_test_split

from skimage.transform import rescale, resize

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, SGD

import glob



# We know that MNIST images are 28 pixels in each dimension.
img_width = 90
img_height = 160

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_width * img_height

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_height, img_width)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_height, img_width, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 4

def checkSize(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def centerCrop(image, crop_size):
    crop_size = checkSize(crop_size)
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def randomCrop(image, crop_size):
    crop_size = checkSize(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

def scaleAugmentation(image, scale_range, crop_size):
    scale_size = np.random.randint(*scale_range)
    # image = imresize(image, (scale_size, scale_size))
    image = resize(image, (scale_size, scale_size))
    image = centerCrop(image, crop_size)
    return image

# scaleAugmentation(inimg, (256, 480), 224)
# imgScaleOut = resize(imgScaleOut, img_shape_full)

images = []
labels = []
for serialized_example in tf.python_io.tf_record_iterator('../dataset/OUTPUT/model.tfrecords'):

    feature_set = { 'image': tf.FixedLenFeature([],tf.string),
                    'label': tf.FixedLenFeature([],tf.int64)
    }
    features = tf.parse_single_example(serialized_example,features=feature_set)
    label = features['label']
    label = tf.Session().run(label)
    image = features['image']
    
    image = tf.Session().run(image)
    
    image = np.frombuffer(image, dtype=np.uint8)
    # image = image.astype(float)

    ll= np.zeros(num_classes)
    ll[label] = 1

    # Original
    images.append(image)
    labels.append(ll) 

    # Reshape de la imagen original
    imgResh = np.reshape(image, img_shape_full)

    # Flip Horizontal
    imgHor = np.fliplr(imgResh)
    imgHorCopy = imgHor
    imgHor = imgHor.flatten()
    images.append(imgHor)
    labels.append(ll)

    # Flip Vertical
    imgVer = np.flipud(imgResh)
    imgVerCopy = imgVer
    imgVer = imgVer.flatten()
    images.append(imgVer)
    labels.append(ll)

    # Ampliacion Original
    imgScaleAug = scaleAugmentation(imgResh, (64, 120), 60)
    imgScaleAug = resize(imgScaleAug, img_shape_full)
    imgScaleAug = imgScaleAug.flatten()
    images.append(imgScaleAug)
    labels.append(ll)

    # Ampliacion Flip Horizontal
    imgScaleAug = scaleAugmentation(imgHorCopy, (64, 120), 60)
    imgScaleAug = resize(imgScaleAug, img_shape_full)
    imgScaleAug = imgScaleAug.flatten()
    images.append(imgScaleAug)
    labels.append(ll)

    # Ampliacion Flip Vertical
    imgScaleAug = scaleAugmentation(imgVerCopy, (64, 120), 60)
    imgScaleAug = resize(imgScaleAug, img_shape_full)
    imgScaleAug = imgScaleAug.flatten()
    images.append(imgScaleAug)
    labels.append(ll)

    # Rotaciones
    for i in range(3):
        imgrot = np.rot90(imgResh, i+1)
        # imgrot = np.reshape(imgrot, img_shape_full)
        imgrot = resize(imgrot, img_shape_full)
        imgrot = imgrot.flatten()
        images.append(imgrot)
        labels.append(ll) 
        


images = np.array(images)
labels = np.array(labels)

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from the test-set.
imgs = images[0:9]

# Get the true classes for those images.
lbls = labels[0:9]
lbls = np.argmax(lbls,axis=1)

# Plot the images and labels using our helper-function above.
plot_images(images=imgs, cls_true=lbls)


images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.1)


#########################################
############MODELO SECUENCIAL############
#########################################

#########################################
############CONSTRUCCION MODELO##########
#########################################

# Comienza la construcción del modelo Keras Sequential.
model = Sequential()

# Agrega una capa de entrada que es similar a un feed_dict en TensorFlow.
# Tenga en cuenta que la forma de entrada debe ser una tupla que contenga el tamaño de la imagen.
model.add(InputLayer(input_shape=(img_size_flat,)))

# La entrada es una matriz aplanada con 784 elementos (img_size * img_size),
# pero las capas convolucionales esperan imágenes con forma (28, 28, 1), por tanto hacemos un reshape
model.add(Reshape(img_shape_full))

# Primera capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Segunda capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Aplanar la salida de 4 niveles de las capas convolucionales
# a 2-rank que se puede ingresar a una capa totalmente conectada 
model.add(Flatten())

# Primera capa completamente conectada  con ReLU-activation.
model.add(Dense(128, activation='relu'))

# Última capa totalmente conectada con activación de softmax
# para usar en la clasificación.
model.add(Dense(num_classes, activation='softmax'))


#########################################
############COMPILACION MODELO###########
#########################################

optimizer = Adam(lr=1e-8)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Entrenamiento del modelo
"""
Le informaremos también del batch size, es decir, de la cantidad de 
imágenes que debe usar en cada iteración. 
Así como del numero de epochs, es decir, de cuantas veces va a recorrer el conjunto entero de datos para entrenar.
"""

# k = 10
# step = math.floor(len(labels)/k)
# for i in range(0,k-1):
#     train_images= general_images[:i*step-1]
#     train_labels= general_labels[:i*step-1]
#     validation_images= general_images[i*step:(i+1)*step-1]
#     validation_labels= general_labels[i*step:(i+1)*step-1]
#     train_images= np.concatenate((train_images, general_images[(i+1)*step:]), axis=0)
#     train_labels= np.concatenate((train_labels, general_labels[(i+1)*step:]), axis=0)
#     model.fit(x=train_images,
#           y=train_labels,
#           epochs=5, batch_size=100,verbose=2) #,validation_split=0.2

model.fit(images_train, labels_train, batch_size=5, epochs=30, verbose=1, validation_split=0.1)

# Evaluación del modelo

result = model.evaluate(images_test, labels_test, verbose=0)

print ('Testing set accuracy:', result[1]) 

# Imprimir perdida y precision
# for name, value in zip(model.metrics_names, result):
#     print(name, value)

# #Imprimir solo precision en forma de porcentaje(%)
# print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


#########################################
############PREDICCION###################
#########################################

imgs = images[0:15]

labels_true = labels[0:15]

cls_true = np.argmax(labels_true,axis=1)

labels_pred = model.predict(x=imgs)

#Pasar clases predecidas a enteros
cls_pred = np.argmax(labels_pred,axis=1)
print("Classes true")
print(cls_true)
print("Prediction")
print(cls_pred)

# labels_pred = model.predict(x=imgs, verbose=1)
# print(labels_pred)