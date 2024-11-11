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
        one_hot_label = np.zeros(10)
        one_hot_label[label] = 1
        mnist_labels.append(one_hot_label)
    return np.array(mnist_data), np.array(mnist_labels)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return x > 0
    
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def forward(self, X):
        self.input = X
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.final_input)
        return self.output
    
    def backward(self, X, y):
        m = y.shape[0]
        output_error = self.output - y
        weights_hidden_output_gradient = np.dot(self.hidden_output.T, output_error) / m
        bias_output_gradient = np.sum(output_error, axis=0, keepdims=True) / m
        
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_output)
        weights_input_hidden_gradient = np.dot(X.T, hidden_error) / m
        bias_hidden_gradient = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        self.weights_hidden_output -= self.lr * weights_hidden_output_gradient
        self.bias_output -= self.lr * bias_output_gradient
        self.weights_input_hidden -= self.lr * weights_input_hidden_gradient
        self.bias_hidden -= self.lr * bias_hidden_gradient
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    
    def accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == labels)

class LearningRateScheduler:
    def __init__(self, initial_lr, patience=5, decay_factor=0.5):
        self.initial_lr = initial_lr
        self.patience = patience
        self.decay_factor = decay_factor
        self.best_val_accuracy = 0
        self.wait = 0
    
    def adjust_lr(self, mlp, val_accuracy):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                mlp.lr *= self.decay_factor
                self.wait = 0

def train_mlp(mlp, train_X, train_y, val_X, val_y, epochs=50, batch_size=64):
    scheduler = LearningRateScheduler(initial_lr=mlp.lr)
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_X))
        train_X, train_y = train_X[indices], train_y[indices]
        
        for i in range(0, len(train_X), batch_size):
            X_batch = train_X[i:i + batch_size]
            y_batch = train_y[i:i + batch_size]
            output = mlp.forward(X_batch)
            mlp.backward(X_batch, y_batch)
        
        train_output = mlp.forward(train_X)
        train_accuracy = mlp.accuracy(train_y, train_output)
        val_output = mlp.forward(val_X)
        val_accuracy = mlp.accuracy(val_y, val_output)
        val_loss = mlp.compute_loss(val_y, val_output)
        
        print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
        
        scheduler.adjust_lr(mlp, val_accuracy)
        
        # if val_accuracy >= 0.95:
        #     print("Reached")
        #     break

train_X, train_y = download_mnist(is_train=True)
val_X, val_y = download_mnist(is_train=False)

train_X, val_X = train_X / 255.0, val_X / 255.0

mlp = MLP(input_size=784, hidden_size=100, output_size=10, learning_rate=0.01)
train_mlp(mlp, train_X, train_y, val_X, val_y, epochs=50, batch_size=64)
