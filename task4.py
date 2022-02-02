import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


model = tf.keras.applications.VGG19(weights='imagenet', include_top=True)


gb_model = tf.keras.Model(
    inputs=[model.inputs],
    outputs=[model.get_layer("block5_conv4").output]
)
# layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
for i in range(len(gb_model.layers) - 1):
    if hasattr(gb_model.layers[1 + i], 'activation') and (gb_model.layers[1 + i].activation == tf.keras.activations.relu or gb_model.layers[1 + i].activation == tf.nn.relu):
        gb_model.layers[1 + i].activation = guidedRelu

img = cv2.imread('/content/dog.jpeg')
image = load_img('/content/dog.jpeg', target_size=(224, 224))
iim = img_to_array(image)
iim = iim.reshape((1, iim.shape[0], iim.shape[1], iim.shape[2]))


with tf.GradientTape(persistent=True) as tape:
    inputs = tf.cast(iim, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)

    output_np = outputs.numpy()
    output_np = output_np.reshape(
        output_np.shape[0] * output_np.shape[1] * output_np.shape[2] * output_np.shape[3])
    top_neurons = output_np[output_np.argsort()[-5:][::-1]]

    for val in top_neurons:
        temp = tf.cast(outputs >= val, outputs.dtype) * outputs
        grads = tape.gradient(temp, inputs)[0]
        plt.figure(0)
        plt.imshow(grads.numpy())
        plt.axis('off')
        plt.show()
