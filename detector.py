import sys

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datasource as ds
import visualization as vs


def main():

    # Fetching data

    students = ds.fetch_students()
    input_dataframe, output_dataframe = ds.create_data_for_nn(students)

    input_matrix = input_dataframe.as_matrix().astype(np.float32)
    output_matrix = output_dataframe.as_matrix().astype(np.float32)

    training_inputs = input_matrix[: round(len(input_dataframe) * 0.8)]
    test_inputs = input_matrix[round(len(input_dataframe) * 0.8):]
    training_outputs = output_matrix[: round(len(output_dataframe) * 0.8)]
    test_outputs = output_matrix[round(len(output_dataframe) * 0.8):]

    # Placeholders

    input_placeholder = tf.placeholder(
        tf.float32, [None, len(input_dataframe.columns)], 'input_placeholder')
    output_placeholder = tf.placeholder(
        tf.float32, [None, len(output_dataframe.columns)], 'output_placeholder')

    # Hyper-parameters

    n_epoch = 10000
    learning_rate = 0.001
    n_hidden_1 = 100
    n_hidden_2 = 100

    # Weights and biases

    weights_1 = tf.Variable(
        tf.random_normal([len(input_dataframe.columns), n_hidden_1]))
    biases_1 = tf.Variable(tf.random_normal([n_hidden_1]))

    weights_2 = tf.Variable(
        tf.random_normal([n_hidden_1, n_hidden_2]))
    biases_2 = tf.Variable(tf.random_normal([n_hidden_2]))

    weights_3 = tf.Variable(
        tf.random_normal([n_hidden_2, len(output_dataframe.columns)]))
    biases_3 = tf.Variable(tf.random_normal([len(output_dataframe.columns)]))

    # Logit creation

    hidden_layer_1 = tf.add(tf.matmul(input_placeholder, weights_1), biases_1)
    sigmoid_1 = tf.nn.sigmoid(hidden_layer_1)
    # sigmoid_1 = tf.nn.dropout(sigmoid_1, 0.5)
    hidden_layer_2 = tf.add(tf.matmul(sigmoid_1, weights_2), biases_2)
    sigmoid_2 = tf.nn.sigmoid(hidden_layer_2)
    # sigmoid_2 = tf.nn.dropout(sigmoid_2, 0.5)
    output_layer = tf.add(tf.matmul(sigmoid_2, weights_3), biases_3)
    output_layer = tf.nn.sigmoid(output_layer)

    # Define loss and optimizer

    # cost = tf.reduce_mean(-1 * tf.reduce_sum(
    #     output_placeholder * tf.log(output_layer), reduction_indices=[1]))

    # cost = tf.reduce_sum(
    #     tf.pow(output_layer - output_placeholder, 2)) / (2 * training_inputs.shape[0])

    cost = tf.losses.mean_squared_error(
        labels=output_placeholder, predictions=output_layer)

    # cost = tf.reduce_mean(tf.square(output_placeholder - output_layer))

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=output_placeholder))

    optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)

    test_losses = []
    training_losses = []

    saver = tf.train.Saver(
        [weights_1, biases_1, weights_2, biases_2, weights_3, biases_3])

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        mode = "test"

        if mode is "train":
            # Training cycle

            # Fit all training data
            for epoch in range(n_epoch):
                feed = {input_placeholder: training_inputs,
                        output_placeholder: training_outputs}
                sess.run(optimizer, feed_dict=feed)

                # Display logs per epoch step
                if (epoch + 1) % 100 == 0:
                    feed = {
                        input_placeholder: training_inputs,
                        output_placeholder: training_outputs
                    }
                    c_training = sess.run(cost, feed_dict=feed)
                    training_losses.append(c_training)
                    
                    feed = {
                        input_placeholder: test_inputs,
                        output_placeholder: test_outputs
                    }
                    c_test = sess.run(cost, feed_dict=feed)
                    test_losses.append(c_test)
                    print("Epoch:", '%04d,' % (epoch + 1), 
                        "Training error=", "{:.9f},".format(c_training), 
                        "Test cost=", "{:.9f}".format(c_test))

            print("Optimization Finished!")

            # Do not forget to increase global step
            saver.save(sess, "checkpoint/model.ckpt", global_step=1)

            # Plot training errors
            x_axis = range(len(training_losses))
            y_axis = training_losses
            plt.plot(x_axis, y_axis)
            plt.title("Training errors")
            plt.show()

            # Plot training errors
            x_axis = range(len(test_losses))
            y_axis = test_losses
            plt.plot(x_axis, y_axis)
            plt.title("Test errors")
            plt.show()

        elif mode is "test":
            saver.restore(sess, "checkpoint/model.ckpt-1")

            '''
            # Orcan
            input_row = ds.create_input_row(
                sex='M',
                age=22,
                address='U',
                famsize='GT3',
                medu=2,
                fedu=4,
                mjob='at_home',
                fjob='services',
                reason='reputation',
                traveltime=0.7,
                studytime=1,
                pstatus='A',
                success=0.4,
                failures=1,
                schoolsup=True,
                famsup=True,
                activities=False,
                nursery=True,
                higher=True,
                internet=True,
                romantic=True,
                famrel=0.9,
                freetime=0.3,
                goout=0.1,
                health=0.75,
                absences=10
            )
            '''
            '''
            # Ali
            input_row = ds.create_input_row(
                sex='M',
                age=22,
                address='U',
                famsize='GT3',
                medu=3,
                fedu=3,
                mjob='at_home',
                fjob='other',
                reason='preference',
                traveltime=0.7,
                studytime=0.75,
                pstatus='A',
                success=0.6,
                failures=0.8,
                schoolsup=True,
                famsup=True,
                activities=False,
                nursery=False,
                higher=True,
                internet=True,
                romantic=False,
                famrel=0.65,
                freetime=0.5,
                goout=0.5,
                health=0.7,
                absences=10
            )
            '''
            
            '''
            # Dido 
            input_row = ds.create_input_row(
                sex='F',
                age=2,
                address='U',
                famsize='LE3',
                medu=0,
                fedu=0,
                mjob='at_home',
                fjob='at_home',
                reason='reputation',
                traveltime=0.2,
                studytime=0.3,
                pstatus='T',
                success=0.4,
                failures=0.3,
                schoolsup=True,
                famsup=True,
                activities=False,
                nursery=False,
                higher=False,
                internet=True,
                romantic=False,
                famrel=0.5,
                freetime=1,
                goout=0.1,
                health=0.8,
                absences=3
            )
            '''
            '''
            # Mahir
            input_row = ds.create_input_row(
                sex='M',
                age=18,
                address='U',
                famsize='GT3',
                medu=3,
                fedu=3,
                mjob='other',
                fjob='other',
                reason='reputation',
                traveltime=0.7,
                studytime=0.1,
                pstatus='A',
                success=0.5,
                failures=0,
                schoolsup=False,
                famsup=True,
                activities=True,
                nursery=False,
                higher=True,
                internet=True,
                romantic=False,
                famrel=0.8,
                freetime=0.8,
                goout=0,
                health=1,
                absences=3
            )
            '''

            '''
            # İlayda
            input_row = ds.create_input_row(
                sex='F',
                age=20,
                address='U',
                famsize='GT3',
                medu=4,
                fedu=2,
                mjob='at_home',
                fjob='other',
                reason='preference',
                traveltime=1,
                studytime=0.8,
                pstatus='A',
                success=0.7,
                failures=1,
                schoolsup=False,
                famsup=True,
                activities=False,
                nursery=False,
                higher=True,
                internet=True,
                romantic=True,
                famrel=0,
                freetime=0.6,
                goout=0.1,
                health=0.7,
                absences=0
            )
            '''
            
            '''
            # Doruk
            input_row = ds.create_input_row(
                sex='M',
                age=22,
                address='U',
                famsize='GT3',
                medu=4,
                fedu=4,
                mjob='other',
                fjob='other',
                reason='preference',
                traveltime=1,
                studytime=0.7,
                pstatus='A',
                success=2,
                failures=1,
                schoolsup=False,
                famsup=True,
                activities=False,
                nursery=True,
                higher=True,
                internet=True,
                romantic=True,
                famrel=0.3,
                freetime=0.5,
                goout=0.3,
                health=0.5,
                absences=12
            )
            '''

            input_matrix = np.array([input_row])

            feed = {
                input_placeholder: input_matrix
            }
            output = sess.run(output_layer, feed_dict=feed)
            print(output)


if __name__ == '__main__':
    main()
