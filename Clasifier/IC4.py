from random import shuffle
from datetime import datetime
import os
import random
import sys
import threading
import glob

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('input_directory', '../dataset/INPUT',
                           'Input data directory')
tf.app.flags.DEFINE_string('output_file', '../dataset/OUTPUT/model.tfrecords',
                           'Output file')
tf.app.flags.DEFINE_string('labels_file', '../dataset/labels_file.txt',
                           'Labels file')
FLAGS = tf.app.flags.FLAGS


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=1)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='grayscale', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return filename.endswith('.png')


def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  assert image.shape[2] == 1

  image = np.squeeze(np.asarray(image))
  # np.array2string(image)
  return image

#REURNS IMAGES AND LABELS ARRAYS
def _find_image_files():
  data_dir = FLAGS.input_directory
  labels_file = FLAGS.labels_file
  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.GFile(
      labels_file, 'r').readlines()]

  labels = []
  images = []

  # Labeling starts from 0.
  label_index = 0
  coder = ImageCoder()

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    
    for path in matching_files:
      images.append(_process_image(path,coder))

    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(images)))
  random.seed(12345)
  random.shuffle(shuffled_index)
  
  print('Image size: %d. Label size: %d.' %(len(images), len(labels)))
  images = [images[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Image size: %d. Label size: %d.' %(len(images), len(labels)))
  print('Found %d JPEG files across %d labels inside %s.' %
        (len(images), len(unique_labels), data_dir))
  return images, labels

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(unused_argv):
  images, labels =_find_image_files()
  print(type(images))
  print(type(labels))
  print('Image size: %d. Label size: %d.' %(len(images), len(labels)))
  
  output_file = FLAGS.output_file  # file to save the TFRecords file
  # open the TFRecords file
  writer = tf.python_io.TFRecordWriter(output_file)
  
  for i in list(range(len(labels))):
    # Create a feature
    feature = {'label': _int64_feature(labels[i]),
               'image': _bytes_feature(tf.compat.as_bytes(np.array2string(images[i])))} #images[i].tostring()
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
  writer.close()
  sys.stdout.flush()
  
if __name__ == '__main__':
  tf.app.run()
