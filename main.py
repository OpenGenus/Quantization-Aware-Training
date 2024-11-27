import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
import numpy as np

# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Apply Quantization Aware Training (QAT) to the model
def apply_quantization_aware_training(model):
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    return quantized_model

# Prepare the dataset
def prepare_data():
    # Use MNIST dataset for example purposes
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, -1).astype(np.float32) / 255.0
    return (x_train, y_train), (x_test, y_test)

# Convert and save the model
def convert_and_save_model(model, filepath):
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the model to a file
    with open(filepath, 'wb') as f:
        f.write(tflite_model)

# Main workflow
if __name__ == "__main__":
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = prepare_data()

    # Create and apply QAT to the model
    model = create_model()
    quantized_model = apply_quantization_aware_training(model)

    # Compile the model
    quantized_model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    # Train the model
    quantized_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Evaluate the model
    loss, accuracy = quantized_model.evaluate(x_test, y_test)
    print(f'Quantized model accuracy: {accuracy:.4f}')

    # Convert and save the model
    convert_and_save_model(quantized_model, 'quantized_model.tflite')

