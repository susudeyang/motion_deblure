import numpy as np
from scipy import fft
from skimage import io, exposure
import cv2
from skimage.util import img_as_ubyte, img_as_float
from tqdm import trange
import matplotlib.pyplot as plt
import argparse
import os




def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)


def toeplitizMatrix(n, row):
    vecDD = np.zeros(n)
    vecDD[0] = 4
    vecDD[1] = -1
    vecDD[row] = -1
    vecDD[-1] = -1
    vecDD[-row] = -1
    return vecDD


def vectorize(matrix):
    return matrix.T.ravel()


def reshape(vector, row, col):
    return vector.reshape((row, col), order='F')


class LIME:
    def __init__(self, iterations=10, alpha=2, rho=2, gamma=0.7, strategy=2, *args, **kwargs):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        self.L = img_as_float(imgPath)
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]
        self.imagepath = imgPath

        self.T_hat = np.max(self.L, axis=2)
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        self.W = self.weightingStrategy()

    def weightingStrategy(self):
        if self.strategy == 2:
            dTv = self.dv @ self.T_hat
            dTh = self.T_hat @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack([Wv, Wh])
        else:
            return np.ones((self.row * 2, self.col))

    def __T_subproblem(self, G, Z, u):
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft.fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        T = np.real(reshape(T, self.row, self.col))
        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __derivative(self, matrix):
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])

    def __AG_cal(self):
        image=self.L.astype(np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        average_gradient = np.mean(gradient_magnitude)

        return average_gradient

    def __contrast_cal(self):
        image = self.L.astype(np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        contrast = np.std(gray)

        return contrast

    def illumMap(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        u = 1

        for _ in trange(0, self.iterations):
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

        return T ** self.gamma

    def enhance(self):
        self.T = self.illumMap()
        self.R = self.L / np.repeat(self.T[:, :, np.newaxis], 3, axis=2)
        self.R = exposure.rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R



    def cal_flag(self):
        AG=self.__AG_cal()
        contrast=self.__contrast_cal()
        self.flag=AG#+contrast


def main(options):
    lime = LIME(**options.__dict__)
    image = io.imread(options.filePath)
    lime.load(image)
    lime.cal_flag()
    print(lime.flag)
    if lime.flag<20:
        lime.enhance()
        filename = os.path.split(options.filePath)[-1]
        if options.output:
            savePath = f"{options.output}enhanced_{filename}"
            plt.imsave(savePath, lime.R)
        #if options.map:
            savePath = f"{options.output}map_{filename}"
            plt.imsave(savePath, lime.T, cmap='gray')
    else:
        print("图像很亮了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filePath", default="./data/381.jpg", type=str, help="image path to enhance")
    #parser.add_argument("-m", "--map", action="store_true", help="save illumination map")
    parser.add_argument("-o", "--output", default="./", type=str, help="output folder")
    options = parser.parse_args()
    main(options)
