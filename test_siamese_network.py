# USAGE : $ python3 test_siamese_network.py -i1 ./img1.jpg -i2 ./img2.jpg

import tensorflow as tf
import numpy as np
from helper import preprocess_image, inference
from scipy.spatial.distance import cdist
import argparse
import os
from PIL import Image
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Test images.')
parser.add_argument('-i1', '--img1', metavar='--img1', type=str, nargs='?', help='Path to image 1')
parser.add_argument('-i2', '--img2', metavar='--img2', type=str, nargs='?', help='Path to image 2')
args = parser.parse_args()

if __name__ == '__main__':
    left_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'left_input_im')
    right_input_im = tf.placeholder(tf.float32, [None, 256, 128, 3], 'right_input_im')
    left_label = tf.placeholder(tf.float32, [None, ], 'left_label')
    right_label = tf.placeholder(tf.float32, [None, ], 'right_label')

    print(np.shape(left_input_im), np.shape(right_input_im))
    logits, model_left, model_right = inference(left_input_im, right_input_im)

    # total_loss = tf.losses.get_total_loss()
    global_step = tf.Variable(0, trainable=False)
    global_init = tf.variables_initializer(tf.global_variables())

    img1 = np.array(Image.open(args.img1))
    img1 = img1/255
    img1 = img1 - 0.5
    img1 *= 2.0
    img1 = np.expand_dims(img1, axis=0)
    
    img2 = np.array(Image.open(args.img2))
    img2 = img2/255
    img2 = img2 - 0.5
    img2 *= 2.0
    img2 = np.expand_dims(img2, axis=0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(global_init)
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model_siamese/model.ckpt")

        my_logits, model_lf, model_rg = sess.run([logits, model_left, model_right], \
                                                 feed_dict={left_input_im: img1, right_input_im: img2})

        print("Logits", my_logits)
        print("Left shape", np.shape(model_lf))
        print("Right shape", np.shape(model_rg))

        lft = np.array(model_lf[0])
        rgt = np.array(model_rg[0])
        l = lft - rgt

        distance = np.sqrt(np.sum((l) ** 2))
        similarity = my_logits * np.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - np.array(my_logits[0])) * np.square(np.max((0.5 - distance),
                                                                        0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = np.mean(dissimilarity + similarity) / 2
        print('distance : ', distance)
        print('similarity : ', similarity)
        print('dissimilarity : ', dissimilarity)
        print('similarity_loss : ', similarity_loss)

        dist = cdist(model_lf, model_rg, 'cosine')
        print('Pairwise distance : ', dist)
        euc = np.linalg.norm(model_lf - model_rg)
        print('euc : ', euc)

        if my_logits > 0.0:
            print('Similar')
        else:
            print('Dissimilar')
