# CSS 490 Final Project Implemntation Part 3
# Names: Luca de Raad and Davis Zhong
# Date: 06/02/2022
# Description: This python file creates and evaluates four pretrained InceptionNetV3 
# TFLite models using the Keras mnist dataset. Models include normal unmodified InceptionV3,
# pruned model of InceptionNetV3, quantized model of InceptionNetV3, and both pruned
# and quantized model of InceptionNetV3. The program is made to be runnable on the Jetson Nano.


# imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time
from pathlib import Path

# load in dataset
mnist = keras.datasets.mnist
(test_images, test_labels) = mnist.load_data()[1]

# image count to control the amount of images tested
max_images = 1000

test_images = test_images[0:max_images, 0:28, 0:28]

test_images = np.expand_dims(test_images, axis=3)
test_images = tf.image.resize(test_images, [75, 75])
test_images = test_images / 255


# runs inferencing on a model and calculates relevant metrics
def evaluate_model(evaluate_interpreter, model_name):
    input_index = evaluate_interpreter.get_input_details()[0]["index"]
    output_index = evaluate_interpreter.get_output_details()[0]["index"]

    # metric variables
    total_time_interpreting = 0
    slowest_inference_time = 0
    fastest_inference_time = 99999

    prediction_digits = []

    # image inferencing loop
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        evaluate_interpreter.set_tensor(input_index, test_image)

        start = time()

        # Run inference.
        evaluate_interpreter.invoke()

        end = time()

        # calculating metrics
        total_time_interpreting += end - start
        fastest_inference_time = min(fastest_inference_time, end - start)
        slowest_inference_time = max(slowest_inference_time, end - start)

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = evaluate_interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)

    num_correct = 0

    for i in range(max_images):
        if prediction_digits[i] == test_labels[i]:
            num_correct += 1

    print(f"Model {model_name} was trained on {max_images}")

    accuracy = num_correct / max_images
    print(f"Top 1 accuracy was {accuracy}")

    print(f"Total time spent was {total_time_interpreting}")

    avg_inference_time = total_time_interpreting / max_images
    print(f"Average inference time for each image was {avg_inference_time}")

    print(f"Fastest inference time for an image was {fastest_inference_time}")
    print(f"Slowest inference time for an image was {slowest_inference_time}")


# Interprets tflite models obtained using a converter
def benchmark_tflite(model_name):
    tflite_models_dir = Path.cwd()

    model = tflite_models_dir / model_name
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()

    print("Beginning evaluation of ", model_name)

    evaluate_model(interpreter, model_name)

    print("Evaluation of ", model_name, " complete!")

    del interpreter


# main driver method
def main():
    benchmark_tflite("normal.tflite")

    benchmark_tflite("pruned.tflite")

    benchmark_tflite("quantized.tflite")

    benchmark_tflite("pruned_quantized.tflite")

    print("done!")


if __name__ == '__main__':
    main()
