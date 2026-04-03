import cv2
import numpy as np
import time

def linear_motion_blur(image, kernel_size, angle):
    """
    线性运动模糊
    :param image: 输入图像
    :param kernel_size: 模糊核大小（运动距离）
    :param angle: 运动角度（0-360度）
    :return: 模糊后的图像
    """
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    angle_rad = np.deg2rad(angle)
    
    # 计算核的中心点
    center = kernel_size // 2
    
    # 计算运动方向上的点
    for i in range(kernel_size):
        x = int(center + center * np.cos(angle_rad))
        y = int(center + center * np.sin(angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # 归一化核
    kernel = kernel / kernel.sum()
    
    # 应用卷积
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def separable_gaussian_blur(image, kernel_size, sigma):
    """
    分离式高斯模糊
    :param image: 输入图像
    :param kernel_size: 模糊核大小
    :param sigma: 高斯分布的标准差
    :return: 模糊后的图像
    """
    # 应用水平方向高斯模糊
    blurred = cv2.GaussianBlur(image, (kernel_size, 1), sigma)
    # 应用垂直方向高斯模糊
    blurred = cv2.GaussianBlur(blurred, (1, kernel_size), sigma)
    return blurred

def wiener_deblur(image, kernel, noise_var=0.001):
    """
    维纳滤波去模糊
    :param image: 模糊图像
    :param kernel: 模糊核
    :param noise_var: 噪声方差估计
    :return: 去模糊后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 获取图像和核的大小
    img_height, img_width = gray.shape
    kernel_height, kernel_width = kernel.shape
    
    # 计算需要填充的大小
    pad_height = img_height - kernel_height
    pad_width = img_width - kernel_width
    
    # 对核进行零填充
    padded_kernel = np.zeros_like(gray)
    padded_kernel[:kernel_height, :kernel_width] = kernel
    
    # 傅里叶变换
    img_fft = np.fft.fft2(gray)
    kernel_fft = np.fft.fft2(padded_kernel)
    
    # 计算功率谱
    kernel_power = np.abs(kernel_fft) ** 2
    
    # 维纳滤波
    wiener_filter = np.conj(kernel_fft) / (kernel_power + noise_var)
    restored_fft = img_fft * wiener_filter
    
    # 逆傅里叶变换
    restored = np.abs(np.fft.ifft2(restored_fft))
    
    # 转换为8位图像
    restored = np.uint8(restored)
    
    return restored

def inverse_filter_deblur(image, kernel):
    """
    快速逆滤波去模糊
    :param image: 模糊图像
    :param kernel: 模糊核
    :return: 去模糊后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 获取图像和核的大小
    img_height, img_width = gray.shape
    kernel_height, kernel_width = kernel.shape
    
    # 对核进行零填充
    padded_kernel = np.zeros_like(gray)
    padded_kernel[:kernel_height, :kernel_width] = kernel
    
    # 傅里叶变换
    img_fft = np.fft.fft2(gray)
    kernel_fft = np.fft.fft2(padded_kernel)
    
    # 避免除以零
    kernel_fft[kernel_fft == 0] = 1e-8
    
    # 逆滤波
    restored_fft = img_fft / kernel_fft
    
    # 逆傅里叶变换
    restored = np.abs(np.fft.ifft2(restored_fft))
    
    # 转换为8位图像
    restored = np.uint8(restored)
    
    return restored

def test_fft(image):
    start_time = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    dft_height = cv2.getOptimalDFTSize(height)
    dft_width = cv2.getOptimalDFTSize(width)
    padding_image=np.zeros(dft_height, dft_width, dtype=float, order='C')
    padding_image[:height, :width] = gray

    cv2.dft(gray,gray,)


# 测试代码
if __name__ == "__main__":
    # 读取测试图像
    image = cv2.imread('405.jpg')
    if image is None:
        print("无法读取图像，请确保test_image.jpg存在")
        exit()
    
    print(f"图像大小: {image.shape}")




    # 测试添加线性运动模糊
    start_time = time.time()
    blurred_linear = linear_motion_blur(image, kernel_size=20, angle=45)
    end_time = time.time()
    print(f"线性运动模糊耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
    
    # 测试添加高斯模糊
    start_time = time.time()
    blurred_gaussian = separable_gaussian_blur(image, kernel_size=15, sigma=3)
    end_time = time.time()
    print(f"分离式高斯模糊耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
    
    # 创建测试用的运动模糊核
    kernel_size = 20
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1
    kernel = kernel / kernel.sum()
    
    # 将图像转换为灰度并添加已知模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    known_blur = cv2.filter2D(gray, -1, kernel)
    
    # 测试维纳滤波去模糊
    start_time = time.time()
    restored_wiener = wiener_deblur(known_blur, kernel)
    end_time = time.time()
    print(f"维纳滤波去模糊耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
    
    # 测试逆滤波去模糊
    start_time = time.time()
    restored_inverse = inverse_filter_deblur(known_blur, kernel)
    end_time = time.time()
    print(f"逆滤波去模糊耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
    
    # 保存结果
    cv2.imwrite('original.jpg', image)
    cv2.imwrite('blurred_linear.jpg', blurred_linear)
    cv2.imwrite('blurred_gaussian.jpg', blurred_gaussian)
    cv2.imwrite('restored_wiener.jpg', restored_wiener)
    cv2.imwrite('restored_inverse.jpg', restored_inverse)
    
    print("处理完成，结果已保存")
