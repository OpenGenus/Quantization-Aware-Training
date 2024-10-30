# Quantization Aware Training
## Project Overview
This project builds a Convolutional Neural Network (CNN) using TensorFlow and applies Quantization Aware Training (QAT) to make it more efficient for edge devices (e.g., mobile or IoT devices). The model is trained on the MNIST dataset, which contains images of handwritten digits, and then converted to a TensorFlow Lite (TFLite) format for optimized deployment.

## Requirements
Install the following packages:

`pip install tensorflow tensorflow_model_optimization numpy`

## Main Steps

- **Load Data:** The MNIST dataset is loaded and prepared for training.
- **Build the Model:** A simple CNN is defined to classify images into 10 digit classes.
- **Apply Quantization:** QAT is applied to make the model smaller and faster without losing much accuracy.
- **Train and Evaluate:** The model is trained on the data, and its accuracy is checked.
- **Convert to TFLite:** The trained model is converted to a TFLite format (quantized_model.tflite) for efficient deployment.

## Running the Code

To run the code, execute: `python3 main.py`

## Output
- **Accuracy:** Displays the model's accuracy after training.
- **Model File:** Saves the quantized model as quantized_model.tflite.

## Authors

**Vidhi Srivastava**

- [GitHub](https://github.com/Vidhi0229)
- [LinkedIn](https://www.linkedin.com/in/vidhisrivastava01/)

## Show Your Support ⭐️⭐️
If you find this project helpful or interesting, please consider giving it a star!


