
import scipy
from scipy.sparse.linalg import LinearOperator
import numpy as np
import cv2
from skimage import io
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
import argparse
import threading
import copy


def initial_d(image):
    px = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    py = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    pxx = cv2.Sobel(px, cv2.CV_64F, 1, 0, ksize=5)
    pyy = cv2.Sobel(py, cv2.CV_64F, 0, 1, ksize=5)
    pxy = cv2.Sobel(px, cv2.CV_64F, 0, 1, ksize=5)
    pyx = cv2.Sobel(py, cv2.CV_64F, 1, 0, ksize=5)
    pmerge = (pxy + pyx)/2
    p = [px, py, pxx, pyy, pmerge]
    return p
class Filter_thres(threading.Thread):
    def __init__(self, mag, a, index, m):
        super(Filter_thres,self).__init__()
        self.mag = mag
        self.a = a
        self.index = index
        self.m = m

    def __sortfilter(self, matrix):
        matrix = np.array(matrix)
        vector = matrix.flatten()
        vector.sort()
        threshold = vector[-self.m]
        return threshold

    def __angle_mask(self):
        threshold_low = self.index * 45
        threshold_high = (self.index + 1) * 45
        _, a_mask = cv2.threshold(self.a, threshold_high, 1, cv2.THRESH_TOZERO_INV)
        _, a_mask = cv2.threshold(a_mask, threshold_low, 1, cv2.THRESH_BINARY)
        m_mask = a_mask * self.mag
        thresh = self.__sortfilter(m_mask)
        _, m_mask = cv2.threshold(m_mask, thresh, 1, cv2.THRESH_BINARY)
        self.m_mask = m_mask
        print("wanbi")

    def run(self):
        self.__angle_mask()

    def download_res(self):
        return self.m_mask

class Deblur:
    def __init__(self, iteration=20, sigmar=2.0, dt=1, ksize=(35, 35), *args, **kwargs):
        self.iteration = iteration
        self.sigmar = sigmar
        self.dt = dt
        self.ksize = ksize
        self.ws = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.m = 2*max(self.ksize[0], self.ksize[1])
        self.beta = 5

    def load(self, imagepath):
        #image = cv2.imread(imagepath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = io.imread(imagepath, as_gray=True)
        self.L = image
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]
        self.pre = copy.deepcopy(self.L)

    def __update_hyp(self):
        self.m *= 2
        self.dt *= 0.9
        self.sigmar *= 0.9
    def __shock_filter(self,image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobel_x**2, sobel_y**2)
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
        signal = np.sign(laplacian)
        shock_image = image-signal*magnitude*self.dt
        return shock_image

    def __magnitude_threshold(self, image):
        mag_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        mag_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        magnitude, angle = cv2.cartToPolar(mag_x, mag_y, angleInDegrees=True)
        angle = angle % 180
        t = []
        for i in range(4):
            temp_thread = Filter_thres(magnitude, angle, i, self.m)
            t.append(temp_thread)
            t[i].setName("t{}".format(i))
            t[i].start()
        t[0].join()
        t[1].join()
        t[2].join()
        t[3].join()
        mask = np.zeros(magnitude.shape, dtype=np.double)
        for i in range(4):
            mask += t[i].download_res()
        dstimage = mask*image
        return dstimage

    def __AAK(self, v):
        kernal = v.reshape(self.ksize, order='C')
        kernal = np.pad(kernal, ((0, self.AAK.shape[0]-self.ksize[0]), (0, self.AAK.shape[1]-self.ksize[1])), mode='constant')
        kernal_fft = np.fft.fft2(kernal)
        AAK = self.AAK * kernal_fft
        AAK = np.real(np.fft.ifft2(AAK))
        AAK = AAK[:self.ksize[0], :self.ksize[1]]
        AAK = AAK + self.beta*kernal[:self.ksize[0], :self.ksize[1]]
        AAK = AAK.flatten(order='C').reshape(-1, 1)
        print(AAK.dtype)
        return AAK.ravel()

    def prediction(self, image):
        image = denoise_bilateral(image, 5, self.sigmar, 2)
        image = self.__shock_filter(image)
        image = self.__magnitude_threshold(image)
        self.pre = copy.deepcopy(image)



    def kernal_estimate(self):
        p = initial_d(self.pre)
        b = initial_d(self.L)#初始化成功
        '''self.kernal = np.zeros(self.ksize)
        center_x, center_y = self.ksize[0]//2, self.ksize[1]//2
        self.kernal[center_x-2:center_x+3, center_y-2:center_y+3] = 1
        self.kernal = self.kernal / np.sum(self.kernal)'''
        wk = self.ksize[1]
        hk = self.ksize[0]
        '''w = cv2.getOptimalDFTSize(self.L.shape[1])
        h = cv2.getOptimalDFTSize(self.L.shape[0])'''
        w = self.L.shape[1]+self.ksize[1]-1
        h = self.L.shape[0]+self.ksize[0]-1
        #beta = np.full([h, w], self.beta)
        for i in range(5):
            p[i] = np.pad(p[i], ((0, h-self.L.shape[0]),(0, w-self.L.shape[1])), mode='constant')
            b[i] = np.pad(b[i], ((0, h-self.L.shape[0]),(0, w-self.L.shape[1])), mode='constant')
            p[i] = np.fft.fft2(p[i])
            b[i] = np.fft.fft2(b[i])
        #beta = np.fft.fft2(beta, [h, w])
        #self.BETA = copy.deepcopy(beta)
        #self.p = p
        #self.kernal = np.fft.fft2(self.kernal, [h,w], norm='ortho')
        AB = np.zeros(b[0].shape, np.complex128)
        AAK = np.zeros(p[0].shape, np.complex128)
        for i in range(5):
            AB += self.ws[i]*p[i].conj()*b[i]
            AAK += self.ws[i] * p[i].conj() * p[i]
        self.AAK = AAK
        AB = np.real(np.fft.ifft2(AB))
        AB = AB[:hk, :wk]#这里剪裁要考虑了
        AB = AB.flatten(order='C').reshape(-1, 1)
        print(AB.shape)
        ATAop = LinearOperator(shape=(self.ksize[0]*self.ksize[1], self.ksize[0]*self.ksize[1]) , matvec = self.__AAK, dtype=np.float64)
        self.kernal, cgs_info = scipy.sparse.linalg.cgs(ATAop, AB, x0=self.kernal.flatten(order='C').reshape(-1, 1), tol = 0.003, maxiter = 100)
        self.kernal = self.kernal.reshape(self.ksize, order='C')
        if cgs_info == 0:
            print(f"✅ cgs迭代收敛成功！迭代次数≤500")
        else:
            print(f"⚠️ cgs迭代未收敛，info={cgs_info}（1=超迭代次数，-1=数值错误）")


    def update_image(self):
        snr = 50
        srcImg = copy.deepcopy(self.L)
        kernal = copy.deepcopy(self.kernal)
        h, w = self.L.shape[0]+self.ksize[0]-1, self.L.shape[1]+self.ksize[0]-1
        srcImg_paded = np.pad(srcImg, ((0, h-srcImg.shape[0]),(0, w-srcImg.shape[1])), mode='constant')
        #offset_h, offset_w = (H-h)//2, (W-w)//2
        #kernel_pad[offset_h:offset_h+h, offset_w:offset_w+w] = kernal / np.sum(kernal)
        H_pad, W_pad = srcImg_paded.shape
        kernel_pad = np.pad(kernal, ((0, h-self.ksize[0]), (0, w-self.ksize[1])), mode='constant')
        kernel_pad = kernel_pad / np.sum(kernel_pad)
        img_fft = np.fft.fft2(srcImg_paded)
        k_fft = np.fft.fft2(kernel_pad)
        rec_fft = (np.conj(k_fft) / (np.abs(k_fft) ** 2 + 1 / snr)) * img_fft
        rec = np.fft.ifft2(rec_fft).real
        rec = rec[:self.L.shape[0], :self.L.shape[1]]
        rec = (rec - rec.min()) / (rec.max() - rec.min()) * 255
        return rec.astype(np.uint8)

    def __show_kernal(self):
        kernal = self.kernal
        kernal = kernal / np.sum(kernal)
        kernal = (kernal - kernal.min()) / (kernal.max() - kernal.min()) * 255
        plt.imshow(kernal, cmap='gray')
        plt.show()
    def enhance(self, image):
        preimage = copy.deepcopy(image)
        for i in range(self.iteration):
            self.prediction(preimage)
            if i > 0:
                pass
            else:
                sigma = 0.3 * ((self.ksize[1] - 1) / 2 - 1) + 0.8
                x, y = np.meshgrid(np.arange(self.ksize[1])-(self.ksize[1]-1)/2,
                                   np.arange(self.ksize[0])-(self.ksize[0]-1)/2)
                self.kernal = np.exp(-(x**2 + y**2) / (2*sigma**2))
                self.kernal = self.kernal / np.sum(self.kernal)
                self.__show_kernal()
            self.kernal_estimate()
            preimage = self.update_image()
            plt.imshow(preimage, cmap='gray')
            plt.show()
            self.__show_kernal()
            self.__update_hyp()
        ##输出最后一轮的preimage
        self.pre = copy.deepcopy(preimage)

def main(options):
    deblure = Deblur(** options.__dict__)
    deblure.load(options.filePath)
    deblure.enhance(deblure.pre)
    plt.imshow(deblure.pre, cmap='gray')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filePath", default="./blur.jpg", type=str, help="image path to enhance")
    parser.add_argument("-o", "--output", default="./test.jpg", type=str, help="output folder")

    parser.add_argument("-i", "--iterations", default=20, type=int, help="iteration number")
    parser.add_argument("-s", "--sigmar", default=2.0, type=int, help="parameter of alpha")
    parser.add_argument("-t", "--dt", default=1.0, type=int, help="parameter of rho")
    parser.add_argument("-k", "--ksize", default=(35, 39), type=int, help="parameter of gamma")
    options = parser.parse_args()
    main(options)