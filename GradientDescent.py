import matplotlib.pyplot as plt
import numpy as np

from DataStorage import DataStorage


class GradientDescent:
    __features = []
    __results = []
    __j_function = []
    __theta = []
    __features_norm = []

    def __init__(self, theta, path):
        data = DataStorage(path)
        self.__features = data.get_data()[:-1]
        self.__features_norm = self.__features
        self.__results = np.array(data.get_data()[1])
        self.__theta = np.array(theta)[np.newaxis].transpose()

    def cost_function(self):
        m = len(self.__results)
        n = len(self.__features[0])
        J = 0
        for i in range(n):
            J += (np.matmul(self.__theta.transpose(), self.__features[0][i]) - self.__results[i]) ** 2
        return J / (2 * m)

    def scale(self):
        for i in range(len(self.__theta)):
            temp_max = np.max(self.__features_norm[0][:, i])
            self.__features_norm[0][:, i] = self.__features_norm[0][:, i] / temp_max
        temp_max = np.max(self.__results)
        self.__results = self.__results / temp_max

    def gradient(self, num_iter, alpha):
        m = len(self.__results)
        n = len(self.__features_norm[0][0])
        # self.scale()
        for iter in range(num_iter):
            temp = np.zeros(len(self.__theta))
            for i in range(n):
                hO = 0
                for j in range(m):
                    hO += (np.matmul(self.__theta.transpose(), self.__features_norm[0][j]) - self.__results[j]) \
                          * self.__features_norm[0][j, i]
                temp[i] = self.__theta[i] - alpha * hO / m

            self.__theta = temp
            self.__j_function.append(self.cost_function())
        return self.__theta

    def plot_j(self, num_iter):
        plt.plot([i for i in range(num_iter)], self.__j_function)
        plt.show()

    def plot_result(self):
        plt.scatter(self.__features[0][:, 1], self.__results)
        plt.plot(self.__features[0], np.matmul(self.__theta.transpose(), self.__features[0].transpose()))
        plt.show()

    def plot_3D(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.__features[0][:, 0], self.__features[0][:, 1],
                np.matmul(self.__theta.transpose(), self.__features[0].transpose()))
        ax.scatter(self.__features[0][:, 0], self.__features[0][:, 1], self.__results, color='r')
        ax.view_init(0, 0)
        plt.show()

    def get_cost(self, X):
        print(np.matmul(np.array(self.__theta), np.array(X)[np.newaxis, :].transpose()))
