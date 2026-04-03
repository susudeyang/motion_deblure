import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class MotionKernelInitializer:
    """运动模糊核初始估计生成器"""

    @staticmethod
    def gaussian_kernel(kernel_shape, sigma=1.0, center=None):
        """
        高斯核 - 适用于相机抖动类模糊
        """
        nk, mk = kernel_shape
        if center is None:
            center = (nk // 2, mk // 2)

        y, x = np.ogrid[:nk, :mk]
        y_center, x_center = center

        kernel = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)

        return kernel

    @staticmethod
    def linear_motion_kernel(kernel_shape, length=None, angle=0):
        """
        线性运动模糊核 - 适用于匀速直线运动
        参数:
            kernel_shape: (height, width)
            length: 运动长度（像素）
            angle: 运动角度（弧度，0表示水平向右）
        """
        nk, mk = kernel_shape

        if length is None:
            length = min(nk, mk) // 2

        # 创建空核
        kernel = np.zeros(kernel_shape)

        # 计算中心
        center_y, center_x = nk // 2, mk // 2

        # 计算终点坐标
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)

        # 从起点到终点画线
        from skimage.draw import line
        start_y = int(round(center_y - dy / 2))
        start_x = int(round(center_x - dx / 2))
        end_y = int(round(center_y + dy / 2))
        end_x = int(round(center_x + dx / 2))

        # 确保在核内
        start_y = max(0, min(start_y, nk - 1))
        start_x = max(0, min(start_x, mk - 1))
        end_y = max(0, min(end_y, nk - 1))
        end_x = max(0, min(end_x, mk - 1))

        # 绘制线段
        try:
            rr, cc = line(start_y, start_x, end_y, end_x)
            # 确保索引在范围内
            valid_idx = (rr >= 0) & (rr < nk) & (cc >= 0) & (cc < mk)
            rr, cc = rr[valid_idx], cc[valid_idx]

            # 设置值
            kernel[rr, cc] = 1.0
            kernel = kernel / np.sum(kernel)
        except:
            # 如果出错，使用简单的中心核
            kernel[center_y, center_x] = 1.0

        return kernel

    @staticmethod
    def uniform_kernel(kernel_shape, radius=None):
        """
        均匀核 - 适用于未知方向的运动
        """
        nk, mk = kernel_shape

        if radius is None:
            # 使用圆形区域
            y, x = np.ogrid[:nk, :mk]
            center_y, center_x = nk // 2, mk // 2
            radius = min(nk, mk) // 4

            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            kernel = np.zeros(kernel_shape)
            kernel[mask] = 1.0
        else:
            # 使用矩形区域
            kernel = np.ones(kernel_shape) / (nk * mk)

        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def curvilinear_kernel(kernel_shape, curve_params=None):
        """
        曲线运动模糊核 - 适用于手抖或复杂运动
        """
        nk, mk = kernel_shape
        kernel = np.zeros(kernel_shape)

        center_y, center_x = nk // 2, mk // 2

        # 默认曲线：正弦波
        if curve_params is None:
            amplitude = min(nk, mk) / 8
            frequency = 2
            num_points = 20

            # 生成曲线点
            t = np.linspace(-np.pi / 2, np.pi / 2, num_points)
            x = center_x + (mk / 4) * t / (np.pi / 2)
            y = center_y + amplitude * np.sin(frequency * t)

            points = list(zip(
                np.clip(y.astype(int), 0, nk - 1),
                np.clip(x.astype(int), 0, mk - 1)
            ))
        else:
            points = curve_params

        # 连接点形成曲线
        for i in range(len(points) - 1):
            y1, x1 = points[i]
            y2, x2 = points[i + 1]

            # 简单的线性插值
            num_steps = max(abs(y2 - y1), abs(x2 - x1)) + 1
            for step in range(num_steps):
                y = int(y1 + (y2 - y1) * step / max(num_steps - 1, 1))
                x = int(x1 + (x2 - x1) * step / max(num_steps - 1, 1))
                if 0 <= y < nk and 0 <= x < mk:
                    kernel[y, x] = 1.0

        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def sparse_kernel(kernel_shape, sparsity=0.1):
        """
        稀疏核 - 适用于CGS快速收敛
        """
        nk, mk = kernel_shape
        kernel = np.zeros(kernel_shape)

        # 中心点
        center_y, center_x = nk // 2, mk // 2
        kernel[center_y, center_x] = 1.0

        # 随机添加一些非零元素
        num_nonzero = int(nk * mk * sparsity)
        indices = np.random.choice(nk * mk, num_nonzero, replace=False)

        for idx in indices:
            y = idx // mk
            x = idx % mk
            kernel[y, x] = np.random.rand()

        # 归一化
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def delta_kernel(kernel_shape):
        """
        脉冲核 - 最简单，但可能不适合严重模糊
        """
        kernel = np.zeros(kernel_shape)
        kernel[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
        return kernel

    @staticmethod
    def hybrid_kernel(kernel_shape, method='auto'):
        """
        混合方法：根据核尺寸自动选择
        """
        nk, mk = kernel_shape

        if method == 'auto':
            # 根据尺寸选择
            if max(nk, mk) / min(nk, mk) > 2:
                # 非常非方形：线性运动
                length = max(nk, mk) // 2
                angle = 0 if nk < mk else np.pi / 2  # 水平或垂直
                return MotionKernelInitializer.linear_motion_kernel(
                    kernel_shape, length, angle
                )
            elif nk * mk > 100:  # 大核
                return MotionKernelInitializer.gaussian_kernel(kernel_shape, sigma=2.0)
            else:  # 小核
                return MotionKernelInitializer.uniform_kernel(kernel_shape, radius=3)

        elif method == 'multi_point':
            # 多中心点，加速收敛
            kernel = np.zeros(kernel_shape)
            centers = [
                (nk // 4, mk // 4),
                (nk // 4, 3 * mk // 4),
                (3 * nk // 4, mk // 4),
                (3 * nk // 4, 3 * mk // 4),
                (nk // 2, mk // 2)
            ]

            for cy, cx in centers:
                if 0 <= cy < nk and 0 <= cx < mk:
                    kernel[cy, cx] = 1.0

            return kernel / np.sum(kernel)


def visualize_kernels(kernel_shapes):
    """可视化不同初始核"""
    methods = [
        ('高斯核', 'gaussian_kernel'),
        ('线性运动核', 'linear_motion_kernel'),
        ('均匀核', 'uniform_kernel'),
        ('曲线核', 'curvilinear_kernel'),
        ('稀疏核', 'sparse_kernel'),
        ('脉冲核', 'delta_kernel'),
        ('混合核', 'hybrid_kernel')
    ]

    fig, axes = plt.subplots(len(methods), len(kernel_shapes),
                             figsize=(4 * len(kernel_shapes), 3 * len(methods)))

    if len(kernel_shapes) == 1:
        axes = axes.reshape(-1, 1)

    for i, (name, method_name) in enumerate(methods):
        for j, (nk, mk) in enumerate(kernel_shapes):
            ax = axes[i, j]

            if method_name == 'gaussian_kernel':
                kernel = MotionKernelInitializer.gaussian_kernel((nk, mk))
            elif method_name == 'linear_motion_kernel':
                kernel = MotionKernelInitializer.linear_motion_kernel(
                    (nk, mk), length=nk // 2, angle=np.pi / 4
                )
            elif method_name == 'uniform_kernel':
                kernel = MotionKernelInitializer.uniform_kernel((nk, mk))
            elif method_name == 'curvilinear_kernel':
                kernel = MotionKernelInitializer.curvilinear_kernel((nk, mk))
            elif method_name == 'sparse_kernel':
                kernel = MotionKernelInitializer.sparse_kernel((nk, mk), sparsity=0.1)
            elif method_name == 'delta_kernel':
                kernel = MotionKernelInitializer.delta_kernel((nk, mk))
            elif method_name == 'hybrid_kernel':
                kernel = MotionKernelInitializer.hybrid_kernel((nk, mk))

            ax.imshow(kernel, cmap='hot', interpolation='nearest')
            ax.set_title(f'{name}\n({nk}×{mk})')
            ax.axis('off')

            # 显示能量集中度
            energy = np.sum(kernel ** 2)
            ax.text(0.05, 0.95, f'E={energy:.3f}',
                    transform=ax.transAxes, color='white', fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.show()

    return fig


# 测试不同核形状
kernel_shapes = [(15, 15), (25, 15), (15, 25), (35, 35)]
fig = visualize_kernels(kernel_shapes)