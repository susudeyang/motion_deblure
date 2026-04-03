import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
def gaussian_kernel(size, sigma, distance_type='euclidean'):
    """
    生成不同距离定义下的高斯模糊核

    参数:
    size: 核的大小 (奇数)
    sigma: 高斯函数的标准差
    distance_type: 距离类型 ('euclidean', 'manhattan', 'chebyshev')

    返回:
    高斯核矩阵
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            # 计算到中心点的距离
            dx = i - center
            dy = j - center

            if distance_type == 'euclidean':
                # 欧几里得距离: sqrt(dx² + dy²)
                distance = np.sqrt(dx ** 2 + dy ** 2)
            elif distance_type == 'manhattan':
                # 曼哈顿距离 (小区距离): |dx| + |dy|
                distance = abs(dx) + abs(dy)
            elif distance_type == 'chebyshev':
                # 棋盘距离: max(|dx|, |dy|)
                distance = max(abs(dx), abs(dy))
            else:
                raise ValueError(f"未知的距离类型: {distance_type}")

            # 应用高斯函数
            kernel[i, j] = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    # 归一化核
    kernel = kernel / np.sum(kernel)
    return kernel


def plot_gaussian_kernels():
    """
    绘制并比较不同距离定义下的高斯模糊核
    """
    # 设置参数
    size = 15  # 核大小
    sigma = 2.0  # 标准差

    # 生成三种距离定义下的高斯核
    kernel_euclidean = gaussian_kernel(size, sigma, 'euclidean')
    kernel_manhattan = gaussian_kernel(size, sigma, 'manhattan')
    kernel_chebyshev = gaussian_kernel(size, sigma, 'chebyshev')

    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

    # 1. 3D曲面图
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')

    # 2. 2D热力图
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # 生成坐标网格
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # 绘制3D曲面
    titles = ['欧几里得距离', '曼哈顿距离 (小区距离)', '棋盘距离']
    kernels = [kernel_euclidean, kernel_manhattan, kernel_chebyshev]
    axes_3d = [ax1, ax2, ax3]
    axes_2d = [ax4, ax5, ax6]

    for i, (ax3d, ax2d, kernel, title) in enumerate(zip(axes_3d, axes_2d, kernels, titles)):
        # 3D曲面
        surf = ax3d.plot_surface(X, Y, kernel, cmap='viridis',
                                 edgecolor='k', alpha=0.8)
        ax3d.set_title(title, fontsize=12, fontweight='bold')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('权重')
        ax3d.view_init(elev=30, azim=45)

        # 2D热力图
        im = ax2d.imshow(kernel, cmap='viridis', interpolation='none')
        ax2d.set_title(f'{title} - 热力图', fontsize=10)
        ax2d.set_xlabel('X')
        ax2d.set_ylabel('Y')

        # 添加数值标注（中心区域）
        center = size // 2
        for x_idx in range(center - 1, center + 2):
            for y_idx in range(center - 1, center + 2):
                value = kernel[x_idx, y_idx]
                ax2d.text(y_idx, x_idx, f'{value:.3f}',
                          ha='center', va='center',
                          color='white' if value > 0.1 else 'black',
                          fontsize=8)

    # 添加颜色条
    plt.colorbar(im, ax=axes_2d, orientation='horizontal', fraction=0.02, pad=0.1)

    plt.suptitle('不同距离定义下的高斯模糊核对比 (size=15, σ=2.0)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return kernel_euclidean, kernel_manhattan, kernel_chebyshev


def analyze_kernel_differences(kernel_euclidean, kernel_manhattan, kernel_chebyshev):
    """
    分析并显示核的差异统计信息
    """
    print("=" * 60)
    print("高斯模糊核差异分析")
    print("=" * 60)

    # 计算差异矩阵
    diff_eu_ma = np.abs(kernel_euclidean - kernel_manhattan)
    diff_eu_ch = np.abs(kernel_euclidean - kernel_chebyshev)
    diff_ma_ch = np.abs(kernel_manhattan - kernel_chebyshev)

    # 打印统计信息
    print("\n1. 核的权重总和 (归一化后应为1):")
    print(f"   欧几里得距离核总和: {np.sum(kernel_euclidean):.6f}")
    print(f"   曼哈顿距离核总和: {np.sum(kernel_manhattan):.6f}")
    print(f"   棋盘距离核总和: {np.sum(kernel_chebyshev):.6f}")

    print("\n2. 中心点权重比较:")
    center = kernel_euclidean.shape[0] // 2
    print(f"   欧几里得距离中心权重: {kernel_euclidean[center, center]:.6f}")
    print(f"   曼哈顿距离中心权重: {kernel_manhattan[center, center]:.6f}")
    print(f"   棋盘距离中心权重: {kernel_chebyshev[center, center]:.6f}")

    print("\n3. 最大差异统计:")
    print(f"   欧几里得 vs 曼哈顿 - 最大差异: {np.max(diff_eu_ma):.6f}")
    print(f"   欧几里得 vs 棋盘 - 最大差异: {np.max(diff_eu_ch):.6f}")
    print(f"   曼哈顿 vs 棋盘 - 最大差异: {np.max(diff_ma_ch):.6f}")

    print("\n4. 平均差异统计:")
    print(f"   欧几里得 vs 曼哈顿 - 平均差异: {np.mean(diff_eu_ma):.6f}")
    print(f"   欧几里得 vs 棋盘 - 平均差异: {np.mean(diff_eu_ch):.6f}")
    print(f"   曼哈顿 vs 棋盘 - 平均差异: {np.mean(diff_ma_ch):.6f}")

    print("\n5. 核的熵 (衡量分散程度):")

    def calculate_entropy(kernel):
        # 移除接近0的值以避免log(0)
        kernel = kernel[kernel > 1e-10]
        return -np.sum(kernel * np.log(kernel))

    print(f"   欧几里得距离核熵: {calculate_entropy(kernel_euclidean):.6f}")
    print(f"   曼哈顿距离核熵: {calculate_entropy(kernel_manhattan):.6f}")
    print(f"   棋盘距离核熵: {calculate_entropy(kernel_chebyshev):.6f}")
    print("   (熵值越小表示权重越集中)")

    print("\n6. 核的有效半径 (包含95%权重的半径):")

    def effective_radius(kernel):
        center = kernel.shape[0] // 2
        sorted_indices = np.argsort(-kernel.flatten())
        cumulative_sum = np.cumsum(kernel.flatten()[sorted_indices])
        return np.where(cumulative_sum >= 0.95)[0][0] / kernel.size * kernel.shape[0]

    print(f"   欧几里得距离有效半径: {effective_radius(kernel_euclidean):.2f} 像素")
    print(f"   曼哈顿距离有效半径: {effective_radius(kernel_manhattan):.2f} 像素")
    print(f"   棋盘距离有效半径: {effective_radius(kernel_chebyshev):.2f} 像素")


def visualize_distance_contours():
    """
    可视化不同距离定义的等值线
    """
    size = 11
    center = size // 2

    # 创建网格
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)

    # 计算不同距离
    euclidean_dist = np.sqrt(X ** 2 + Y ** 2)
    manhattan_dist = np.abs(X) + np.abs(Y)
    chebyshev_dist = np.maximum(np.abs(X), np.abs(Y))

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    distances = [euclidean_dist, manhattan_dist, chebyshev_dist]
    titles = ['欧几里得距离', '曼哈顿距离', '棋盘距离']

    for ax, dist, title in zip(axes, distances, titles):
        # 绘制等高线
        contours = ax.contour(X, Y, dist, levels=10, colors='black', alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8)

        # 填充颜色
        im = ax.imshow(dist, cmap='viridis', origin='lower',
                       extent=[-center, center, -center, center])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

        # 标记中心点
        ax.plot(0, 0, 'ro', markersize=8, label='中心点')
        ax.legend()

    plt.suptitle('不同距离定义的等值线图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    print("开始生成和比较不同距离定义下的高斯模糊核...")

    # 可视化距离等值线
    print("\n1. 可视化不同距离定义的等值线:")
    visualize_distance_contours()

    # 生成并可视化高斯核
    print("\n2. 生成不同距离定义下的高斯模糊核:")
    kernels = plot_gaussian_kernels()

    # 分析差异
    print("\n3. 分析核之间的差异:")
    analyze_kernel_differences(*kernels)

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("1. 欧几里得距离核: 权重随圆形等值线递减，最符合物理直觉")
    print("2. 曼哈顿距离核: 权重随菱形等值线递减，计算效率高")
    print("3. 棋盘距离核: 权重随正方形等值线递减，最扩散")
    print("\n三种距离定义产生不同的权重分布，证明距离定义对高斯模糊核有显著影响。")