import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from GradientDescent import GradientDescent

num_inter = 2000
alpha = 0.01
gradient = GradientDescent(np.array([0, 0]), 'Data/ex1data1.txt')
print(gradient.gradient(num_inter, alpha))
gradient.plot_j(num_inter)
gradient.plot_result()
gradient.get_cost([1, 20.341])
