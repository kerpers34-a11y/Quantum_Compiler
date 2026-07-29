"""量子环境与状态(QuantumEnvironment)。

负责:量子态(state_vector + density_matrix 双份同步)、经典/通用寄存器、
内存、PC/LR/SF/ZF、噪声模型的应用与 Kraus 信道。
由 evaluator.py 拆包而来(见 xqishell/evaluator/__init__.py)。
"""

import numpy as np

from xqishell import config
from xqishell.errors import XQIExecutionError

from .noise import build_raw_kraus_ops


class QuantumEnvironment:
    def __init__(self, qreg_size=0, creg_size=0, max_registers=config.MAX_REGISTER,
                 max_memory=config.MAX_MEMORY, simulation_mode='statevector', seed=None):
        # 基础参数
        if not all(isinstance(x, int) and x >= 0 for x in [qreg_size, creg_size]):
            raise XQIExecutionError("Register sizes must be non-negative integers")
        self.qreg_size = qreg_size
        self.creg_size = creg_size
        self.simulation_mode = simulation_mode.lower()
        self.max_registers = max_registers
        self.max_memory = max_memory
        # 实例级随机源:测量采样等都经由它,替代原先对全局 np.random 的依赖
        # (import 本模块不再重置全局随机流;seed 固定时全链路可复现)
        self.rng = np.random.default_rng(seed)
        # 惰性状态的显式声明(此前在方法内动态挂载,靠 hasattr/getattr 防御)
        self.creg_ideal = None      # 首次测量时创建:理想(SV)测量结果
        self.measured_bits = set()  # 本 shot 内已测量的量子位
        self.shot_completed = False
        # 初始化量子寄存器大小和经典寄存器大小
        self._initial_qreg_size = qreg_size
        self._initial_creg_size = creg_size
        # 初始化量子态
        self.initial_state_vector = None
        self.initial_density_matrix = None
        self._reset_quantum_register(qreg_size)
        self.state_vector = np.array([], dtype=np.complex128)
        self.density_matrix = np.array([], dtype=np.complex128)
        self.quantum_state = self.density_matrix
        # 初始化经典寄存器
        self._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.creg = self._initial_creg.copy()
        # 初始化通用寄存器和存储
        self.registers = np.zeros(max_registers, dtype=np.float64)
        self.memory = np.zeros(max_memory, dtype=np.float64)
        # 控制寄存器
        self.pc = 0 # 程序计数器
        self.lr = 0 # 链接寄存器
        self.SF = 0 # 符号标志（负数）
        self.ZF = 0 # 零标志
        # 错误模型
        self.error_model = (config.DEFAULT_Q_ERROR_CODE, config.DEFAULT_Q1_ERROR_PROBABILITY, config.DEFAULT_Q2_ERROR_PROBABILITY, config.DEFAULT_MEASURE_ERROR_PROBABILITY, config.DEFAULT_RESET_ERROR_PROBABILITY)
        self._pending_error_model = (config.DEFAULT_Q_ERROR_CODE, config.DEFAULT_Q1_ERROR_PROBABILITY, config.DEFAULT_Q2_ERROR_PROBABILITY, config.DEFAULT_MEASURE_ERROR_PROBABILITY, config.DEFAULT_RESET_ERROR_PROBABILITY)
        self.initial_mode = simulation_mode.lower() # 记录初始设定的模式
        self.simulation_mode = self.initial_mode

    def _reset_quantum_register(self, new_size):
        """这是改变寄存器大小的唯一来源"""
        self.qreg_size = new_size
        dim = 2 ** new_size if new_size > 0 else 1

        # 创建理想态备份
        sv = np.zeros(dim, dtype=np.complex128)
        sv[0] = 1.0 + 0j
        self.initial_state_vector = sv
        self.state_vector = self.initial_state_vector.copy()

        # 创建物理态备份
        dm = np.zeros((dim, dim), dtype=np.complex128)
        dm[0, 0] = 1.0 + 0j
        self.initial_density_matrix = dm
        self.density_matrix = self.initial_density_matrix.copy()

        # 重要：保持旧变量指向 density_matrix，防止其他地方报错
        self.quantum_state = self.density_matrix
    def resize_qreg(self, new_size):
        """指令调用 qreg q[n] 时触发"""
        self._reset_quantum_register(new_size)

    def reset_for_shot(self):
        """每一轮 Shot 开始时调用"""
        # 检查备份是否存在，防止 NoneType 报错
        if self.initial_state_vector is not None:
            self.state_vector = self.initial_state_vector.copy()
        if self.initial_density_matrix is not None:
            self.density_matrix = self.initial_density_matrix.copy()
            self.quantum_state = self.density_matrix

        self.creg = self._initial_creg.copy()
        self.pc = 0
        self.lr = 0
        self.SF = 0
        self.ZF = 0
        self.shot_completed = False
        self.measured_bits = set()
    def full_reset(self):
        """完全重置所有状态（用于环境初始化）"""
        self.reset_for_shot()
        self.registers.fill(0.0)
        self.memory.fill(0.0)
    def convert_to_density(self):
        """把模拟模式切换为 density_matrix(应用噪声/局部误差前调用)。

        密度矩阵在双模式下始终维护,这里只需翻转模式标志。
        """
        self.simulation_mode = 'density_matrix'

    def apply_quantum_noise(self, qubits):
        if self.qreg_size == 0 or not qubits:
            return
        if not self.error_model:
            return

        noise_type, p1_data, p2_data, p_measure, p_reset = self.error_model

        # 确定使用单比特还是双比特参数（原逻辑扩展）
        params = p1_data if len(qubits) == 1 else p2_data
        # 如果 params 不是列表，则封装成列表以适配 generate_kraus_operators
        if not isinstance(params, list):
            params = [params]

        self.convert_to_density()
        for qubit in qubits:
            kraus_ops = self.generate_kraus_operators(noise_type, qubit, params)
            self.apply_kraus_channel(kraus_ops)

    def generate_kraus_operators(self, noise_type, qubit_idx, params):
        """构造并提升 Kraus 算符到全系统空间(原始算符由 noise 模块纯函数构造)。"""
        raw_ops = build_raw_kraus_ops(noise_type, params)
        return [self.lift_operator(m, qubit_idx) * coeff for coeff, m in raw_ops]

    def lift_operator(self, single_op, target_qubit):
        """把单比特算符提升到全系统空间(q[n-1] ⊗ ... ⊗ q[0],高位在左)。"""
        full_op = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.qreg_size - 1, -1, -1):
            full_op = np.kron(full_op, single_op if i == target_qubit else np.eye(2))
        return full_op

    def get_full_operator(self, op, target_qubit):
        """将单比特算符扩展到全系统空间，匹配 C 语言的高位在左原则"""
        return self.lift_operator(op, target_qubit)

    def apply_kraus_channel(self, kraus_ops):
        """对密度矩阵应用 Kraus 信道 ρ→ΣKρK†,并在迹非零时归一化。"""
        new_rho = np.zeros_like(self.density_matrix, dtype=np.complex128)
        for k in kraus_ops:
            new_rho += k @ self.density_matrix @ k.conj().T
        trace = np.real(np.trace(new_rho))
        self.density_matrix = new_rho / trace if trace > 1e-15 else new_rho
        self.quantum_state = self.density_matrix
    def apply_unitary(self, u_matrix):
        """同时作用于两个数组，并确保引用同步"""
        self.state_vector = u_matrix @ self.state_vector
        self.density_matrix = u_matrix @ self.density_matrix @ u_matrix.conj().T
        self.quantum_state = self.density_matrix # 同步引用

    def apply_depolarizing_error(self, qubit, p):
        """错误只作用于密度矩阵"""
        if p <= 0: return
        X = self.get_full_operator(np.array([[0, 1], [1, 0]], dtype=np.complex128), qubit)
        Y = self.get_full_operator(np.array([[0, -1j], [1j, 0]], dtype=np.complex128), qubit)
        Z = self.get_full_operator(np.array([[1, 0], [0, -1]], dtype=np.complex128), qubit)

        rho = self.density_matrix
        self.density_matrix = (1 - p) * rho + (p / 3.0) * (
                X @ rho @ X.conj().T + Y @ rho @ Y.conj().T + Z @ rho @ Z.conj().T
        )
        self.quantum_state = self.density_matrix

    def apply_physical_reset(self, target_qubit, p_reset):
        """对密度矩阵应用带噪声的重置"""
        # 此处使用你原本的逻辑，但目标改为 self.density_matrix
        # 1. 构造 Trace 算符
        m0 = np.array([[1, 0]], dtype=np.complex128)
        m1 = np.array([[0, 1]], dtype=np.complex128)

        f0 = self.lift_operator(m0, target_qubit)
        f1 = self.lift_operator(m1, target_qubit)
        # 计算偏迹后的 rho
        rho_reduced = f0 @ self.density_matrix @ f0.conj().T + \
                      f1 @ self.density_matrix @ f1.conj().T

        # 2. 注入带噪声的 |0>
        r_matrix = np.array([
            [1.0 - p_reset, np.sqrt((1.0 - p_reset) * p_reset)],
            [np.sqrt(p_reset * (1.0 - p_reset)), p_reset]
        ], dtype=np.complex128)

        self.density_matrix = self._reinsert_qubit(rho_reduced, r_matrix, target_qubit)
        self.quantum_state = self.density_matrix

        # 3. 理想态同步重置（无噪声，直接强制到 0）
        dim = 2 ** self.qreg_size
        mask = ~((np.arange(dim) >> target_qubit) & 1).astype(bool)
        self.state_vector[~mask] = 0
        norm = np.linalg.norm(self.state_vector)
        if norm > 0: self.state_vector /= norm
    def _reinsert_qubit(self, rho_reduced, r_matrix, target_qubit):
        """辅助函数：将缩减后的密度矩阵与新比特矩阵在指定位置重新合并"""
        dim_reduced = 2 ** (self.qreg_size - 1)
        # 构造一个新的 dim x dim 矩阵
        new_rho = np.zeros((dim_reduced * 2, dim_reduced * 2), dtype=np.complex128)
        # 遍历所有基矢，将新比特插入到 target_qubit 位置
        for i in range(dim_reduced):
            for j in range(dim_reduced):
                val = rho_reduced[i, j]
                if abs(val) < 1e-15: continue
                # 将原始索引 i 拆开，在 target_qubit 处插入空位
                # 例如 qreg=3, target=1: i=10 (binary) -> 1_0 (inserted) -> 100, 101, 110, 111...
                def insert_bit(idx, bit):
                    mask = (1 << target_qubit) - 1
                    return ((idx & ~mask) << 1) | (bit << target_qubit) | (idx & mask)
                for b1 in [0, 1]:
                    for b2 in [0, 1]:
                        row = insert_bit(i, b1)
                        col = insert_bit(j, b2)
                        # r_matrix 提供了新比特在该位置的密度矩阵分布
                        new_rho[row, col] = val * r_matrix[b1, b2]
        return new_rho

