import tensorflow as tf
import numpy as np
import pdb

def pre_trained_model(left_im, right_im, reuse=False):
    # Resnet model tensors have changed. Not compatible with checkpoint tensors.
    '''
    from tensorflow.contrib.slim.nets import resnet_v1
    import tensorflow.contrib.slim as slim

    # Create graph
    inputs = tf.placeholder(tf.float32, shape=[64, input.shape[1].value, \
                                            input.shape[2].value, input.shape[3].value])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=False)
    saver = tf.train.Saver()    

    with tf.Session() as sess:
        pdb.set_trace()
        saver.restore(sess, tf.train.latest_checkpoint('resnet50_ckpt'))
        representation_tensor = sess.graph.get_tensor_by_name('feature_generator/resnet_v1_50/pool5:0')
        #img = ...  #load image here with size [1, 224,224, 3]
        #pdb.set_trace()
        features = sess.run(representation_tensor, {'Placeholder:0': input})
    '''
    import tensornets as nets
    # tf.disable_v2_behavior()  # for TF 2

    left_feats = nets.ResNet50(left_im, is_training=True, reuse=tf.AUTO_REUSE)
    left_feats.pretrained()
    right_feats = nets.ResNet50(right_im, is_training=True, reuse=tf.AUTO_REUSE)
    right_feats.pretrained()
    
    merged_features = tf.abs(tf.subtract(left_feats, right_feats))
    return merged_features, left_feats, right_feats
    
def cnn_model(input, reuse=False):
    print(np.shape(input))
    with tf.name_scope("cnn_model"):
        with tf.variable_scope("conv1") as scope:
            net = tf.contrib.layers.conv2d(input, 64, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)

            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 96, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [3, 3], activation_fn=None, padding='SAME',
                                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                           scope=scope, reuse=reuse)
            print(np.shape(net))
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='valid')
            net = tf.layers.batch_normalization(net, fused=True)
            net = tf.nn.relu(net)
            print(np.shape(net))

        net = tf.contrib.layers.flatten(net)
        print(np.shape(net))

        net = tf.layers.dense(net, 4096, activation=tf.sigmoid)
        print(np.shape(net))

    return net
