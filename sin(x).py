import numpy as np
from matplotlib import pyplot as plt


class SinXModel():
    def __init__(self):
        self.epochs = 500000
        self.lr = 0.00005
        self.interval = 10000
        self.x, self.y = self.data_init(100)
        self.paras = np.random.rand(6)
        self.data = self.data_loader(self.x)

    def data_init(self, n):
        x = np.linspace(-np.pi, np.pi, n)
        y = np.sin(x)
        return x, y

    def data_loader(self, x):
        return np.vstack([x ** i for i in range(5, -1, -1)]).T

    def Loss(self, x, y):
        predictions = self.data @ self.paras
        return np.mean((predictions - y) ** 2)

    def train(self):
        for epoch in range(self.epochs):
            output = self.data @ self.paras
            error = output - self.y
            grad = (self.data.T @ error) / len(self.y)
            self.paras -= self.lr * grad

            if epoch % self.interval == 0:
                print(self.Loss(self.x, self.y))

    def plt_draw(self):
        plt.title("sin(x) fit")
        plt.scatter(self.x, self.y, label='Original Data', color='blue')

        y_predict = self.data @ self.paras
        plt.plot(self.x, y_predict, 'r', label='Fitted Line')

        plt.xlabel('x')
        plt.ylabel('Polynomial Fit of sin(x) using Gradient Descent')
        plt.legend()  # Add this line to include the legend
        plt.show()


# Instantiate and train model
model = SinXModel()
model.train()
model.plt_draw()
print(model.paras)
