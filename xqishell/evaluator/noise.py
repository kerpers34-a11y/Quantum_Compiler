"""Kraus 算符构造(纯函数,不依赖量子态)。

支持 6 种噪声模型:
1 去极化 / 2 幅度阻尼 / 3 相位阻尼 / 4 热弛豫 / 5 Pauli / 6 相干幺正。
提升到全系统空间由 QuantumEnvironment.lift_operator 完成。
"""

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def build_raw_kraus_ops(noise_type, params):
    """构造单比特 Kraus 原始算符序列 [(系数, 2x2 矩阵), ...](未提升)。

    params 语义随 noise_type: 1-3 取 params[0] 为概率;
    4 为 [T1, T2, Tgate];5 为 [px, py, pz];6 为 [ex, ey, ez](旋转弧度)。
    """
    p = params[0]
    if noise_type == 1:  # Depolarizing
        return [(np.sqrt(1 - p), I2), (np.sqrt(p / 3), PAULI_X),
                (np.sqrt(p / 3), PAULI_Y), (np.sqrt(p / 3), PAULI_Z)]

    if noise_type == 2:  # Amplitude Damping
        k0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=np.complex128)
        k1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=np.complex128)
        return [(1.0, k0), (1.0, k1)]

    if noise_type == 3:  # Phase Damping
        k0 = np.sqrt(1 - p) * I2
        k1 = np.sqrt(p) * np.array([[1, 0], [0, 0]], dtype=np.complex128)
        k2 = np.sqrt(p) * np.array([[0, 0], [0, 1]], dtype=np.complex128)
        return [(1.0, k0), (1.0, k1), (1.0, k2)]

    if noise_type == 4:  # Thermal Relaxation (热弛豫);params: [T1, T2, Tgate]
        t1, t2, tg = params[0], params[1], params[2]
        if t2 > 2 * t1: t2 = 2 * t1  # 物理约束限制

        p_reset = 1 - np.exp(-tg / t1)
        p_phase = 1 - np.exp(-tg / t2)

        # 组合算符：振幅衰减 + 相位变换(常见近似实现)
        k0 = np.array([[1, 0], [0, np.sqrt(1 - p_phase)]], dtype=np.complex128)
        k1 = np.array([[0, np.sqrt(p_reset)], [0, 0]], dtype=np.complex128)
        # 保持迹守恒的修正项
        k2 = np.array([[np.sqrt(1 - p_reset) - np.sqrt(1 - p_phase), 0], [0, 0]], dtype=np.complex128)
        return [(1.0, k0), (1.0, k1), (1.0, k2)]

    if noise_type == 5:  # Pauli Error (泡利误差);params: [px, py, pz]
        px, py, pz = params[0], params[1], params[2]
        p_id = 1.0 - px - py - pz
        return [(np.sqrt(p_id), I2), (np.sqrt(px), PAULI_X),
                (np.sqrt(py), PAULI_Y), (np.sqrt(pz), PAULI_Z)]

    if noise_type == 6:  # Coherent Unitary Error (相干幺正);params: [ex, ey, ez]
        ex, ey, ez = params[0], params[1], params[2]
        # 构造误差旋转矩阵 U = exp(-i * (ex*X + ey*Y + ez*Z) / 2)
        from scipy.linalg import expm
        u_err = expm(-0.5j * (ex * PAULI_X + ey * PAULI_Y + ez * PAULI_Z))
        return [(1.0, u_err)]

    return [(1.0, I2)]  # Default Identity
