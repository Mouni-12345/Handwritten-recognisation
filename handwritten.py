import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(data_loader):
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(name, data_loader):
    print(f"\n📊 Training on {name} dataset...\n")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess(data_loader)

    model = build_model()
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test)
    print(f"{name} Test Accuracy: {acc:.2f}")

    predictions = model.predict(x_test)
    index = np.random.randint(0, len(x_test))

    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"{name} - Predicted: {np.argmax(predictions[index])}, True: {y_test[index]}")
    plt.axis('off')
    plt.show()

# Run training and evaluation for MNIST and Fashion MNIST
train_and_evaluate("MNIST", mnist)
train_and_evaluate("Fashion MNIST", fashion_mnist)
