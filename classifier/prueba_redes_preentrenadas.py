import numpy as np
import math
from tensorflow.python import keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.python.keras.layers import Input, Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
import glob
import os


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
img_shape_full = (img_height, img_width, 3)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 4


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

    print("salida buffer")
    print(image)
    
    ll= np.zeros(num_classes)
    ll[label] = 1
    labels.append(ll) 
    images.append(image)

images = np.array(images)
labels = np.array(labels)

print(len(images))
# np.set_printoptions(threshold=np.nan)
# print(images[0])
# print(type(images[0]))

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)

# Cargar modelo preentrenado
inp = Input(shape=img_shape_full)
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_shape_full, input_tensor=inp)

# Anadir capa para nuestro numero de clases
x = model.output 
x = GlobalAveragePooling2D()(x) 
predictions = Dense(num_classes, activation='softmax')(x) 
model = Model(inputs=model.input, outputs=predictions)


LAYERS_TO_FREEZE=700
for layer in model.layers[:LAYERS_TO_FREEZE]:
    layer.trainable = False

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)

print ('Testing set accuracy:', score[1]) 