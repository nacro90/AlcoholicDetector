import tensorflow as tf
import numpy as np

import datasource as ds

def main():
    tf.reset_default_graph()
    

    saver = tf.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "checkpoint/model.ckpt")


if __name__ == '__main__':
    main()
