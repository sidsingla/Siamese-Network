import tensorflow as tf
import os
import glob
import numpy as np
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)    
    image = tf.reshape(image, tf.stack([256, 128,3]))
    return image

def get_features(example):
    features = {'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=1),
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
	    	'image/height': tf.FixedLenFeature([], tf.int64),
	    	'image/width': tf.FixedLenFeature([], tf.int64),
	    	'image/format': tf.FixedLenFeature((), tf.string, default_value=""),
                'image/filename': tf.FixedLenFeature((), tf.string, default_value="")}
        
    parsed_features = tf.parse_single_example(example, features)
    image = parsed_features['image/encoded']
    image = tf.image.decode_jpeg(image,channels=3)
    
    # Converts image values from range [0, 255] to [-1, 1]
    image = preprocess_image(image)
    return image, parsed_features['image/class/label'], parsed_features['image/filename']

def make_single_dataset(tfrecords_path, image_size=[256, 128], shuffle_buffer_size=2000, repeat=True, train=True):
    image_size = tf.cast(image_size, tf.int32)        
    filenames = [tfrecords_path]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(12000)
    dataset = dataset.map(get_features, num_parallel_calls=8)
    return dataset

def combine_dataset(data_path, batch_size, image_size, same_prob, diff_prob, repeat=True, train=True):
    """
    same_prob (float): probability of retaining images in same class
    diff_prob (float): probability of retaining images in different class
    repeat (boolean): repeat elements in dataset
    """
    dataset_left = make_single_dataset(data_path, image_size, repeat=repeat, train=train)
    dataset_right = make_single_dataset(data_path, image_size, repeat=repeat, train=train)

    dataset = tf.data.Dataset.zip((dataset_left, dataset_right))

    if train:
        filter_func = create_filter_func(same_prob, diff_prob)
        dataset = dataset.filter(filter_func)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def create_filter_func(same_prob, diff_prob):
    def filter_func(left, right):
        _, right_label, _ = left
        _, left_label, _ = right

        label_cond = tf.equal(right_label, left_label)

        different_labels = tf.fill(tf.shape(label_cond), diff_prob)
        same_labels = tf.fill(tf.shape(label_cond), same_prob)

        weights = tf.where(label_cond, same_labels, different_labels)
        random_tensor = tf.random_uniform(shape=tf.shape(weights))

        return weights > random_tensor

    return filter_func

# Function Implementation https://stackoverflow.com/questions/41172500/how-to-implement-metrics-learning-using-siamese-neural-network-in-tensorflow
# https://github.com/ardiya/siamesenetwork-tensorflow/blob/1dce5933f347eab014954cf5f1e69ff472d14857/model.py#L43
def contrastive_loss(model1, model2, left_label, right_label):
    margin = 0.2
    label = tf.equal(left_label, right_label)
    y = tf.to_float(label)

    with tf.name_scope("contrastive_loss"):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
        similarity_loss = tf.reduce_mean(dissimilarity + similarity) / 2
    return similarity_loss

def inference(left_img, right_img, pre_trained=False):
    with tf.variable_scope('feature_generator', reuse=tf.AUTO_REUSE) as sc:
        if not pre_trained:
            from model import cnn_model
            left_features = cnn_model(tf.layers.batch_normalization(left_img))
            right_features = cnn_model(tf.layers.batch_normalization(right_img))
            merged_features = tf.abs(tf.subtract(left_features, right_features))
        else:
            from model import pre_trained_model
            merged_features, left_features, right_features = pre_trained_model(left_img, right_img)
    logits = tf.layers.dense(merged_features, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
    logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features

def ce_loss(model, label):
    logits = tf.contrib.layers.fully_connected(model, num_outputs=625,
                                               activation_fn=None)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return loss
    
def mod_ce_loss(logits, left_label, right_label):
    label = tf.equal(left_label, right_label)
    label_float = tf.cast(label, tf.float64)

    logits = tf.cast(logits, tf.float64)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_float)) + \
    tf.cast(tf.losses.get_regularization_loss(), tf.float64)

    return loss



