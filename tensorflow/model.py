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


#########################################
############Variables globales###########
#########################################

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
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 2

#########################################
############Leer Dataset#################
#########################################


# Creates a dataset that reads all of the examples from two files(exemple de path="/data/file2.tfrecord").
filenames = ["../dataset/OUTPUT/train-00000-of-00001"]
# filenames = ["../dataset/OUTPUT/train-00000-of-00001", "../dataset/OUTPUT/validation-00000-of-00001"]
dataset = tf.data.TFRecordDataset(filenames)

def _parse_function(example_proto):
    keys_to_features = {'image/encoded': tf.VarLenFeature(tf.string),
                        'image/class/label': tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    image = parsed_features['image/encoded']
    label = parsed_features['image/class/label']
    return [image, label]
# Parse the record into tensors.
dataset = dataset.map(_parse_function)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
x1 = np.array(next_element[0])
x2 = np.array(next_element[1])

while True:
    data_record = next_element
    print("hola")
    print(data_record[0])
    print(data_record[1])

    

# data_path = '../dataset/OUTPUT/train-00000-of-00001'

# feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
#                'image/label': tf.FixedLenFeature([], tf.int64)}
# # Create a list of filenames and pass it to a queue
# filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
# # Define a reader and read the next record
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# # Decode the record read by the reader
# features = tf.parse_single_example(serialized_example, features=feature)
# # Convert the image data from string back to the numbers
# image = tf.decode_raw(features['image/encoded'], tf.float32)

# # Cast label data into int32
# label = tf.cast(features['image/label'], tf.int32)

# print(label)
# print(image)
    # Reshape image data into the original shape
    # image = tf.reshape(image, [224, 224, 3])
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
model.fit(x=dataset.get('image'),
          y=dataset.get('label'),
          epochs=1, batch_size=128, steps=2)

#Evaluación del modelo
result = model.evaluate(x=data.test.images,
                        y=data.test.labels)

#Imprimir perdida y precision
for name, value in zip(model2.metrics_names, result):
    print(name, value)

#Imprimir solo precision en forma de porcentaje(%)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))


#########################################
############PREDICCION###################
#########################################

images = data.test.images[0:9]

cls_true = data.test.cls[0:9]

y_pred = model.predict(x=images)

#Pasar clases predecidas a enteros
cls_pred = np.argmax(y_pred,axis=1)
print(cls_pred)
print(cls_true)

