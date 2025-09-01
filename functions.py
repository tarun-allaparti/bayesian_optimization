import numpy as np

class functions:
    @staticmethod
    def sphere(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -1 * (x1**2 + x2**2) + 100

    @staticmethod
    def branin(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -1 * ((x2 - (5.1 / (4 * (np.pi)**2) * (x1**2)) + (5/(np.pi)) * x1 - 6)**2 + \
               10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    @staticmethod
    def rosenbrock(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -1* 100 * (x2 - x1**2)**2 + (x1 - 1)**2 + 100

    @staticmethod
    def mccormick(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -1*(np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1)
