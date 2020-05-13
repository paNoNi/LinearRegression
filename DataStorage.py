import numpy as np


class DataStorage:
    __X = np.zeros((1, 1))
    __Y = np.zeros((1, 1))

    def __init__(self, path):
        self.read_file(path)
        return

    def read_file(self, path):
        data = []
        with open(path, 'r') as storage:
            for line in storage:
                line = '1' + ',' + line.replace('\n', '')
                data.append([float(num) for num in line.split(',')])
        data = np.array(data)
        self.__X = data[:, 0: -1]
        self.__Y = data[:, -1][np.newaxis, :].transpose()

    def get_data(self):
        return [self.__X, self.__Y]
