import tensorflow as tf
import os
import glob
import numpy as np
import argparse
from helper import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Training options.')
parser.add_argument('-p', '--path', type=str, nargs='?', help='Path to Training data')
parser.add_argument('--pre_trained', type=bool, default=False, help='Use pretrained ResNet50 model from TensorNets? False by Default')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size to use')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train the model for')
args = parser.parse_args()

data_path = args.path
val_data_num = 127233
training_data_num = 382681

# Tensorboard setup
if not os.path.exists('train.log'):
    os.makedirs('train.log')

if __name__ == "__main__":
    BATCH_SIZE = args.batch_size
    num_epochs = args.epochs
    
    train_dataset= combine_dataset(data_path + '/train_00000-of-00001.tfrecord',
                                    batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=True)
    val_dataset = combine_dataset(data_path + '/validation_00000-of-00001.tfrecord', batch_size=BATCH_SIZE, image_size=[256, 128], same_prob=0.5, diff_prob=0.5, train=False)
    handle = tf.placeholder(tf.string, shape=[])

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()

    left_train, right_train = train_iterator.get_next()
    left_im_train, left_label_train, left_file_train = left_train
    right_im_train, right_label_train, right_file_train = right_train

    logits_train, model_left, model_right = inference(left_im_train, right_im_train, args.pre_trained)

    '''
    # Contrastive Loss
    train_similarity_loss = contrastive_loss(model_left, model_right, left_label_train, right_label_train)

    # Cross Entropy loss for classification performance improvement.
    ce_loss_1 = ce_loss(model_left, left_label_train)
    ce_loss_2 = ce_loss(model_right, right_label_train)
    train_ce_loss = tf.add(ce_loss_1, ce_loss_2)
    train_total_loss = tf.add(train_similarity_loss, train_ce_loss)
    '''

    # Modified CE Loss ICML 2015 paper - http://www.cs.toronto.edu/~rsalakhu/papers/oneshot1.pdf
    train_total_loss = mod_ce_loss(logits_train, left_label_train, right_label_train)

    left_val, right_val = val_iterator.get_next()
    left_im_val, left_label_val, left_file_val = left_val
    right_im_val, right_label_val, right_file_val = right_val

    logits_val, model_left_val, model_right_val = inference(left_im_val, right_im_val, pre_trained=args.pre_trained)
    val_total_loss = mod_ce_loss(logits_val, left_label_val, right_label_val)

    global_step = tf.Variable(0, trainable=False)

    params = tf.trainable_variables()
    gradients = tf.gradients(train_total_loss, params)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    updates = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)

    global_init = tf.variables_initializer(tf.global_variables())

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(global_init)

        ## Values for Tensorboard 
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('Training Loss', train_total_loss)

        # To get images in [0, 255] range
        l_im = tf.divide(left_im_train, 2)
        r_im = tf.divide(right_im_train, 2)
        l_im = tf.add(l_im, 0.5)
        r_im = tf.add(r_im, 0.5)
        l_im = tf.multiply(l_im, 255)
        r_im = tf.multiply(r_im, 255)

        tf.summary.image("Left Training data", l_im, max_outputs=BATCH_SIZE)
        tf.summary.image("Right Training data", r_im, max_outputs=BATCH_SIZE)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)
        ##
        
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())

        training_num_iterations = training_data_num // BATCH_SIZE
        val_num_iterations = val_data_num // BATCH_SIZE
        
        for epoch in range(num_epochs):
            print('epoch : ', epoch, ' / ', num_epochs)
            for iteration in range(training_num_iterations):
                feed_dict_train = {handle:training_handle}

                model_l, model_r, left_im, left_file, right_file, logits_, left_lab_train, right_lab_train, loss_train, _, summary_str = \
                sess.run([ model_left, model_right, left_im_train, left_file_train, right_file_train, logits_train, left_label_train,
                                                                  right_label_train, train_total_loss, updates, merged], feed_dict_train)
                #pdb.set_trace()
                writer.add_summary(summary_str, epoch)
                print("iteration : %d / %d - Loss : %f" % (iteration, training_num_iterations, loss_train))

            print( "Validating for epoch : %d" % epoch )
            for val_iter in range(val_num_iterations):
                feed_dict_val = {handle: validation_handle}
                val_loss = sess.run([val_total_loss], feed_dict_val)
                print("iteration : %d / %d - Validation Loss : %f" % (val_iter, val_num_iterations, val_loss[0]))

            if not os.path.exists("./siamese_model/"):
                os.makedirs("./siamese_model/")
            saver.save(sess, "./siamese_model/model.ckpt")
