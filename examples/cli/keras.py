"""
This example is similar to hello_world.py, except that it uses 
a Keras model instead of PyTorch. To try this example with the
benchit cli, run the following command:

benchit keras.py

You should see data for the keras_model instance printed to the
screen.
"""

import tensorflow as tf

tf.random.set_seed(0)

# Define model class
class SmallKerasModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, output_size):
        super(SmallKerasModel, self).__init__()
        self.dense = tf.keras.layers.Dense(output_size, activation="relu")

    def call(self, x):  # pylint: disable=arguments-differ
        output = self.dense(x)
        return output


# Instantiate model and generate inputs
batch_size = 1
input_size = 10
output_size = 5
keras_model = SmallKerasModel(output_size)

inputs = {"x": tf.random.uniform((batch_size, input_size), dtype=tf.float32)}

keras_outputs = keras_model(**inputs)

# Print results
print(f"keras_outputs: {keras_outputs}")