import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

def normalize_data(X):
    return X / 255.0

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def get_batches(X, Y, batch_size=100):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]

def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = softmax(Z)
    return A

def backward_propagation(X, Y, A, W, b, learning_rate=0.1):
    m = X.shape[0]
    dZ = A - Y
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0) / m

    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def train(X_train, Y_train, W, b, epochs=500, learning_rate=0.1, batch_size=100):
    for epoch in range(epochs):
        for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
            A_train = forward_propagation(X_batch, W, b)
            Y_train_encoded = one_hot_encode(Y_batch, 10)
            W, b = backward_propagation(X_batch, Y_train_encoded, A_train, W, b, learning_rate)
        
        if epoch % 5 == 0:
            A_train_full = forward_propagation(X_train, W, b)
            accuracy = compute_accuracy(A_train_full, Y_train)
            print(f"Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%")
    return W, b

def compute_accuracy(A, Y):
    predictions = np.argmax(A, axis=1)
    return np.mean(predictions == Y)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

train_X = normalize_data(train_X)
test_X = normalize_data(test_X)

np.random.seed(42)
W = np.random.randn(784, 10) * 0.01
b = np.zeros(10)

W, b = train(train_X, train_Y, W, b, epochs=50, learning_rate=0.1, batch_size=100)

A_test = forward_propagation(test_X, W, b)
test_accuracy = compute_accuracy(A_test, test_Y)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
