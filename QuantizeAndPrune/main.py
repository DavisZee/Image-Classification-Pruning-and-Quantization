# CSS 490 Final Project Implemntation Part 3
# Names: Luca de Raad and Davis Zhong
# Date: 06/02/2022
# Description: This python file builds and trains InceptionNetV3 models using 
# the Keras MNIST dataset. Then it prunes, quantizes, and converts 
# the models and saves them as TFLite models.
import tempfile
from pathlib import Path

from tensorflow.python.ops.numpy_ops import np_config
import tensorflow.keras as keras
import tensorflow as tf
# from tensorflow import keras
import numpy as np
import tensorflow_model_optimization as tfmot

tflite_models_dir = Path.cwd()

input_shape = (75, 75, 1)


def main():
    np_config.enable_numpy_behavior()
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels) = mnist.load_data()[0]

    # Normalize the input image so that each pixel value is between 0 and 1.
    train_images = train_images / 255.0

    train_images = np.expand_dims(train_images, axis=3)
    train_images = tf.image.resize(train_images, [75, 75])


    # Define the model architecture.
    model = tf.keras.applications.InceptionV3(weights=None, input_shape=input_shape, classes=10)

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_images,
        train_labels,
        epochs=4,
        validation_split=0.1,
    )

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1  # 10% of training set will be used for validation set.

    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    output_model = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    output_model.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    # output_model.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    output_model.fit(train_images, train_labels,
                          batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                          callbacks=callbacks)

    converter_base = tf.lite.TFLiteConverter.from_keras_model(model)
    normal_tflite_model = converter_base.convert()

    save_model(normal_tflite_model, 'normal.tflite', False)

    save_model(normal_tflite_model, 'quantized.tflite', True)

    model_for_export = tfmot.sparsity.keras.strip_pruning(output_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    save_model(pruned_tflite_model, 'pruned.tflite', False)

    save_model(pruned_tflite_model, 'pruned_quantized.tflite', True)


def save_model(model, model_name, quantize):
    # We want to save our files to the other directory
    
    redirect = '../Benchmark'

    tflite_file = tflite_models_dir / redirect / model_name

    if quantize:
        model.converter.optimizations = [tf.lite.Optimize.DEFAULT]

    with open(tflite_file, 'wb') as f:
        f.write(model)

    print(f'Saved {model_name} TFLite model to:', tflite_file)


if __name__ == '__main__':
    main()
