# %matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np
import math

## Import the keras API
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten


import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

import glob


images = []
labels = []
for serialized_example in tf.python_io.tf_record_iterator('../dataset/OUTPUT/model.tfrecords'):
    # example = tf.train.Example()
    # example.ParseFromString(serialized_example)
    # im = example.features.feature['image'].bytes_list.value
    # print(im)
    # print(type(im))
    # im2 = np.fromstring(im, dtype=int)
    # im2 = np.fromstring(im, dtype='<f4')
    # im2 = np.frombuffer(im, dtype=np.uint8)
    # images = np.append(images,np.array(im2))
    # print(type(example.features.feature['label'].int64_list.value))
    # images = np.append(images,np.array(example.features.feature['image'].int64_list.value))
    # labels = np.append(labels,np.array(example.features.feature['label'].int64_list.value))
    # print(type(labels[0]))

    feature_set = { 'image': tf.FixedLenFeature([],tf.string),
                    'label': tf.FixedLenFeature([],tf.int64)
    }
    features = tf.parse_single_example(serialized_example,features=feature_set)
    label = features['label']
    label = tf.Session().run(label)
    image = features['image']
    
    image = tf.Session().run(image)
    
    '''
    image = image.decode() #bytes->string
    image = np.fromstring(image,dtype=np.uint8)#string->array
    '''
    image = np.frombuffer(image, dtype=np.uint8)
    
    ll= np.zeros(2);
    ll[label] = 1;
    labels.append(ll) 
    images.append(image)

images = np.array(images)
labels = np.array(labels)

#data = tf.SparseTensor(images=images, labels=labels)

# if (images.ndim == 1):
#     images = np.array([images])
# if (labels.ndim == 1):
#     labels = np.array([labels])
    
np.set_printoptions(threshold=np.inf)
# print(images[0])
print(labels[0])

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
num_classes = 2


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

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Entrenamiento del modelo
"""
Le informaremos también del batch size, es decir, de la cantidad de 
imágenes que debe usar en cada iteración. 
Así como del numero de epochs, es decir, de cuantas veces va a recorrer el conjunto entero de datos para entrenar.
"""

model.fit(x=images,
          y=labels,
          epochs=1, batch_size=5) 

#Evaluación del modelo
result = model.evaluate(x=images,
                        y=labels)

#Imprimir perdida y precision
for name, value in zip(model.metrics_names, result):
    print(name, value)

#Imprimir solo precision en forma de porcentaje(%)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


#########################################
############PREDICCION###################
#########################################
'''
images = images[0:9]

cls_true = dataset.cls[0:1]

y_pred = model.predict(x=images)

#Pasar clases predecidas a enteros
cls_pred = np.argmax(y_pred,axis=1)
print(cls_pred)
print(cls_true)
'''
