import numpy as np

# -------------------------- 1. 初始化参数（m=3示例，贴合之前设定）--------------------------
n = 5  # 导数图像尺寸（5×5）
m = 3  # 核尺寸（3×3）
pad_size = 8  # 零填充尺寸（8×8，质数幂2³，满足pad_size ≥ n+m-1=7）
omega = 1.0  # 导数图像权重
beta = 0.0   # 正则化参数β

# 构造5×5稀疏导数图像P₊（边缘=1，内部=0，贴合论文稀疏性）
P = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
], dtype=np.float32)

# 构造3×3模糊核K（示例核，可自定义）
K = np.array([
    [0.1, 0.2, 0.1],
    [0.2, 0.4, 0.2],
    [0.1, 0.2, 0.1]
], dtype=np.float32)
K_vec = K.flatten(order='C')  # 核展开为9×1向量

# 核移位与图像移位的对应关系（m=3，t=0~8）
shift_map = [
    (0, 0),  # t=0: 核(0,0) → 图像不移位
    (0, -1), # t=1: 核(0,1) → 图像左移1列
    (0, -2), # t=2: 核(0,2) → 图像左移2列
    (-1, 0), # t=3: 核(1,0) → 图像上移1行
    (-1, -1),# t=4: 核(1,1) → 图像上移1行+左移1列
    (-1, -2),# t=5: 核(1,2) → 图像上移1行+左移2列
    (-2, 0), # t=6: 核(2,0) → 图像上移2行
    (-2, -1),# t=7: 核(2,1) → 图像上移2行+左移1列
    (-2, -2) # t=8: 核(2,2) → 图像上移2行+左移2列
]

# -------------------------- 2. 时域计算：显式构造A矩阵 → 计算AᵀA K --------------------------
def shift_image(img, dx, dy, pad_val=0):
    """图像移位（dx：行移位，dy：列移位），超出边界填充pad_val"""
    h, w = img.shape
    shifted = np.full_like(img, pad_val, dtype=np.float32)
    # 计算有效区域
    src_x = slice(max(0, -dx), min(h, h - dx))
    src_y = slice(max(0, -dy), min(w, w - dy))
    dst_x = slice(max(0, dx), min(h, h + dx))
    dst_y = slice(max(0, dy), min(w, w + dy))
    shifted[dst_x, dst_y] = img[src_x, src_y]
    return shifted

# 构造A矩阵（25行×9列：每行=像素，每列=移位后的P的列向量）
A = []
for (dx, dy) in shift_map:
    # 图像移位
    P_shifted = shift_image(P, dx, dy)
    # 列优先展开为25×1向量，作为A的一列
    A_col = P_shifted.flatten(order='C')  # order='F'：列优先
    A.append(A_col)
A = np.array(A).T  # A: 25×9矩阵

# 计算AᵀA K（时域）
A_T_A = A.T @ A  # 9×9矩阵
A_T_A_K_time = (A_T_A @ K_vec).reshape(m, m)  # 加入正则化βK，重塑为3×3

# -------------------------- 3. 频域计算：按论文步骤实现AᵀA K --------------------------
# 3.1 预处理：P₊零填充
P_pad = np.zeros((pad_size, pad_size), dtype=np.float32)
P_pad[:n, :n] = P  # 左上角填充原始P，其余为0

# 3.2 频域预计算：F(P₊)、复共轭、S
F_P = np.fft.fft2(P_pad)  # 8×8复矩阵
conj_F_P = np.conj(F_P)   # 复共轭
S = omega * conj_F_P * F_P  # 8×8实矩阵（逐元素点积）

# 3.3 核的预处理：零填充+FFT
K_pad = np.zeros((pad_size, pad_size), dtype=np.float32)
K_pad[:m, :m] = K  # 左上角填充原始K，其余为0
F_K = np.fft.fft2(K_pad)  # 8×8复矩阵

# 3.4 频域点积：F(AᵀA K) = S ⊙ F(K)
F_A_T_A_K = S * F_K

# 3.5 逆变换+裁剪
A_T_A_K_fft_raw = np.fft.ifft2(F_A_T_A_K).real  # 取实部消除数值误差
# 裁剪：从8×8中提取3×3核心区域（对应核尺寸）
#A_T_A_K_fft = A_T_A_K_fft_raw[:m, :m]
start_x = 1
start_y = 1
A_T_A_K_fft = A_T_A_K_fft_raw[start_x:start_x+m, start_y:start_y+m]
# 加入正则化βK
A_T_A_K_fft = A_T_A_K_fft

# 3.6 后处理：归一化（论文要求）
max_val = np.max(A_T_A_K_fft)
#A_T_A_K_fft[A_T_A_K_fft < max_val / 20] = 0  # 小于最大值1/20的元素置零
A_T_A_K_fft = A_T_A_K_fft / np.sum(A_T_A_K_fft)  # 归一化求和=1

# 时域结果同样归一化（保持对比公平）
max_val_time = np.max(A_T_A_K_time)
#A_T_A_K_time[A_T_A_K_time < max_val_time / 20] = 0
A_T_A_K_time = A_T_A_K_time / np.sum(A_T_A_K_time)

# -------------------------- 4. 结果对比：验证一致性 --------------------------
print("时域计算结果（AᵀA K_time）：")
print(np.round(A_T_A_K_time, 4))
print("\n频域计算结果（AᵀA K_fft）：")
print(np.round(A_T_A_K_fft, 4))
print("\n两者均方误差（MSE）：", np.mean((A_T_A_K_time - A_T_A_K_fft) ** 2))
print("一致性验证：", "通过" if np.mean((A_T_A_K_time - A_T_A_K_fft) ** 2) < 1e-6 else "失败")