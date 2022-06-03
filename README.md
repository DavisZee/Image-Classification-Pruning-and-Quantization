# Image Classification Pruning and Quantization
### Names: Luca de Raad, Davis Zhong
# Benchmark Data
## Experiment Setup and How the Data was Obtained
In our experiment we used modified versions of the Keras InceptionNetV3 model retrained on the Keras MNIST dataset. We developed four final models being the normal model, quantized model, pruned model, and pruned and quantized model. These models were then converted to TFLite so that they were compatible with the Jetson Nano. Each model was then evaluated on 1000 images from the MNIST testing dataset that were resized to 75x75x1 to fit our models. We then used a loop to test each individual image, making sure to record different metrics such as accuracy and inference time. The results were then saved and printed to the console for further analysis.

### Figure 1.1
![benchmarking data](/Artifacts/benchmark_data.png)
## Discussion and Analysis
We noticed that after quantization, the size of the model demonstrated a 74% decrease in size changing from 87,296,576 to 22,244,576 bytes while pruning made no impact on the model size. Moreover, quantization cut down the average inference time for individual images by over 46% and also nearly halved the overall inference time during testing while only at a cost of 0.2% accuracy. Pruning the model reduced the slowest inference time by 34% while only reducing the average by 0.95%, a similar trend across both the normal and quantized TFLite models. From this we can determine our method of global pruning–setting 50% of the weights in each layer to 0–largely affects the process of inferencing complex or difficult images. Due to the little overall speedup from the quantized model, we can infer that these overly long inference times are outliers to our data. To prove this theory we will likely need to graph all the inference times. From our resulting data we can see that quantizing a model does indeed drop its accuracy, however by also using pruning, we can raise the accuracy, thereby reducing the model’s size, speeding up the model’s overall inference time, and maintain the model’s accuracy.

# Implementation Details
### Figure 1.2
![project diagram](/Artifacts/project_diagram.jpg)
Our project is implemented in two programs. The first program builds, quantizes, and prunes the model, then converts it into TFLite models which can be later used on the Jetson Nano in the second program. The code used to quantize and prune the model is Python (3.9.12) and the most up-to-date version of Tensorflow (2.8.1). This code works by creating a base InceptionNetV3 model, training it on the Keras MNIST dataset, saving it as a TFLite model, creating a quantized TFLite model and then also saving that one. It then prunes the base model, creates a pruned TFLite model, then takes the pruned model and quantizes it into a TFLite model and saves the new pruned and quantized TFLite model.

The second program is used to test and benchmark the TFLite models. Due to the limits of the Jetson Nano, this program uses an earlier version of python (3.6.9) and tensorflow (2.4.1). The benchmarking code in the program takes a subset of the MNIST dataset and resizes it, and records, calculates, and outputs data on each model’s inference time.

# Limitations and Improvements
In this experiment, we were limited by:
- The hardware used to train the models which may have affected the training of the models. (We used a GTX 1060 with 6 GB of memory).
- The data we used to train and test the models. Image labeling could be low quality due to the model only taking in images of 75x75x1.
- The temperature of the hardware. The program had the prepared TFLite models and the benchmarking was conducted after making sure the Jetson Nano was cool. However, we conducted all the benchmarking tests in one go, which means that the temperature of Nano was hotter in the later tests. This could have potentially impacted hardware performance when doing inferences. 
- The images used to test the models. The images were selected from the first 1000 images in the Keras MNIST dataset. They were then resized to be 75x75 because that was InceptionNetV3’s minimum input size from keras. Both the amount of images tested and the selection of them could have affected the benchmark values we collected.

Improvements that could be made to the experiment include:
- Graphing each individual inference times to visually see trends of our quantization and pruning.
- Add a cooldown period between benchmarking TFLite models to ensure that the temperature of the Jetson Nano hardware remains identical at the beginning of the test for each model.
- Increase and improve the set used to train the models. We can load the dataset in batches to allow for more images to be tested without crashing and we can select more random and diverse images from the dataset so that it is not testing the first 1000.
- Do other experimental quantization methods.
- Try local or fine tune pruning.
- We can try different pruning algorithms using different pruning weights.

