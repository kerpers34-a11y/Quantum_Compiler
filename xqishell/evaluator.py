import os
import struct
import numpy as np
from xqishell import config
from xqishell.ast_nodes import ASTNode  # noqa: F401  # re-export,兼容 `from xqishell.evaluator import ASTNode`
from xqishell.errors import XQIExecutionError, XQIInstructionError

# 兼容旧名(旧的 InstructionError(ValueError) 由错误体系中的等价类取代)
InstructionError = XQIInstructionError


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
        p = params[0]
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        raw_ops = []
        if noise_type == 1: # Depolarizing
            raw_ops = [(np.sqrt(1 - p), I), (np.sqrt(p / 3), X), (np.sqrt(p / 3), Y), (np.sqrt(p / 3), Z)]
        elif noise_type == 2: # Amplitude Damping
            k0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=np.complex128)
            raw_ops = [(1.0, k0), (1.0, k1)]
        elif noise_type == 3: # Phase Damping
            k0 = np.sqrt(1 - p) * I
            k1 = np.sqrt(p) * np.array([[1, 0], [0, 0]], dtype=np.complex128)
            k2 = np.sqrt(p) * np.array([[0, 0], [0, 1]], dtype=np.complex128)
            raw_ops = [(1.0, k0), (1.0, k1), (1.0, k2)]
        elif noise_type == 4:  # Thermal Relaxation (热弛豫)
            # params: [T1, T2, Tgate]
            t1, t2, tg = params[0], params[1], params[2]
            if t2 > 2 * t1: t2 = 2 * t1  # 物理约束限制

            p_reset = 1 - np.exp(-tg / t1)
            p_phase = 1 - np.exp(-tg / t2)

            # 组合算符：振幅衰减 + 相位变换
            # 这里采用常见的近似实现
            k0 = np.array([[1, 0], [0, np.sqrt(1 - p_phase)]], dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p_reset)], [0, 0]], dtype=np.complex128)
            # 保持迹守恒的修正项
            k2 = np.array([[np.sqrt(1 - p_reset) - np.sqrt(1 - p_phase), 0], [0, 0]], dtype=np.complex128)
            raw_ops = [(1.0, k0), (1.0, k1), (1.0, k2)]

        elif noise_type == 5:  # Pauli Error (泡利误差)
            # params: [px, py, pz]
            px, py, pz = params[0], params[1], params[2]
            p_id = 1.0 - px - py - pz
            raw_ops = [(np.sqrt(p_id), I), (np.sqrt(px), X), (np.sqrt(py), Y), (np.sqrt(pz), Z)]

        elif noise_type == 6:  # Coherent Unitary Error (相干幺正误差)
            # params: [eps_x, eps_y, eps_z] (微小旋转弧度)
            ex, ey, ez = params[0], params[1], params[2]
            # 构造误差旋转矩阵 U = exp(-i * (ex*X + ey*Y + ez*Z) / 2)
            from scipy.linalg import expm
            u_err = expm(-0.5j * (ex * X + ey * Y + ez * Z))
            raw_ops = [(1.0, u_err)]

        else:
            raw_ops = [(1.0, I)] # Default Identity
        # 提升到全系统空间
        operators = []
        for coeff, m in raw_ops:
            operators.append(self.lift_operator(m, qubit_idx) * coeff)
        return operators
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

class Evaluator:
    def __init__(self, env, parser, ast):
        self.shot_idx = None
        self.shot_total = 0 # 初始化 shot 总数
        self.shots_count_sv = [] # 统计 StateVector 结果
        self.shots_count_dm = [] # 统计 DensityMatrix 结果
        self.env = env
        self.labels = {}
        self.body_instructions = []
        self.source_code_text = parser.get_source_code_text(ast)
        self.labels_info = parser.get_labels_info(ast)
        self.parser = parser
        self.state_dat_path = ""
        self.dm_state_dat_path = ""
        # 调试文件状态(此前在方法内动态挂载,靠 hasattr 防御)
        self.paths = {}
        self._debug_file_initialized = False
        # 每个 shot 捕获第一次测量前的状态(用于最终展示)
        self.pre_measure_state_sv = None

    def evaluate(self, ast):
        # --- 1. 预解析 (Shot, Labels, Instructions) ---
        self.body_instructions = []
        self.labels = {}

        # 提取 Shot 次数
        for node in ast.children:
            if node.type == 'Instruction':
                opcode = next((c.value for c in node.children if c.type == 'Opcode'), None)
                if opcode == 'shot':
                    ops = next((c for c in node.children if c.type == 'Operands'), None)
                    if ops and ops.children: self.shot_total = int(ops.children[0].value)

        # 构建指令体并记录 Label 地址
        for node in ast.children:
            if node.type == 'Label':
                self.labels[node.value.strip(':').strip()] = len(self.body_instructions)
            elif node.type == 'Instruction':
                opcode = next((c.value for c in node.children if c.type == 'Opcode'), None)
                if opcode in ['qreg', 'creg']:
                    self.execute_instruction(node)  # 立即初始化寄存器
                elif opcode and opcode not in ['shot', 'error', 'XQI-BEGIN', 'XQI-END', ';']:
                    self.body_instructions.append(node)

        num_states = 2 ** self.env.qreg_size
        self.shots_count_sv = [0] * num_states
        self.shots_count_dm = [0] * num_states
        self._prepare_final_state_files()  # 初始化结果文件 (.dat)
        self._initialize_debug_files()

        # --- 2. 核心执行循环 (Shot 循环) ---
        for shot_nth in range(1, self.shot_total + 1):
            if shot_nth == 1:
                print(f"Total Program Row:{len(self.body_instructions):-10d}\n")

            self.shot_idx = shot_nth
            self.env.reset_for_shot()  # 内含 measured_bits 重置
            self.pre_measure_state_sv = None  # 每个 shot 重置，捕获第一次测量前的状态

            while self.env.pc < len(self.body_instructions) and not self.env.shot_completed:
                # 打印 PC 轨迹，严格匹配 C 输出：PC=%-10d (shot: %d)
                print(f"PC={self.env.pc:<10d} (shot: {shot_nth})")

                instr_node = self.body_instructions[self.env.pc]
                opcode = next(c.value for c in instr_node.children if c.type == 'Opcode')

                # 捕获测量前状态（用于最后展示 Complete Measure Info）
                if opcode == 'measure' and self.pre_measure_state_sv is None:
                    self.pre_measure_state_sv = self.env.state_vector.copy()

                # 执行指令
                old_pc = self.env.pc
                self.execute_instruction(instr_node)

                # PC 控制逻辑：如果指令没有进行跳转（如 B/BL），则 PC + 1
                if self.env.pc == old_pc:
                    self.env.pc += 1

            # Shot 结束后统计(未发生测量时,理想计数回落到物理 creg)
            creg_sv = self.env.creg_ideal if self.env.creg_ideal is not None else self.env.creg
            self.shots_count_sv[self._calculate_state_code(creg_sv)] += 1
            self.shots_count_dm[self._calculate_state_code(self.env.creg)] += 1
            self._append_shot_states_to_binary()
            print("")

            # --- 3. 最终控制台输出 (匹配 C 语言顺序) ---
        # 打印指令清单
        print("\nOperate Instructions:\n")
        print("XQI-BEGIN")
        # 这里的 source_code_text 建议从 parser 获取原始带行号的文本
        lines = self.source_code_text.splitlines()
        for idx, line in enumerate(lines):
            if any(x in line for x in ["XQI-BEGIN", "XQI-END", "shot", "qreg", "creg", "error"]): continue
            print(f"{line}")
        print("XQI-END\n")

        # 打印标签表
        print("Label Number   Sequence   Label Symbol")
        for idx, (seq, symbol) in enumerate(self.labels_info):
            print(f"Label {idx:3d}:     {seq:3d}        {symbol}")
        print("\n")

        # 打印测量事件信息
        if self.pre_measure_state_sv is not None:
            self._print_complete_measure_info(self.pre_measure_state_sv, filter_unmeasured=False)
            print("\nConsider the qubits NOT measured:")
            self._print_complete_measure_info(self.pre_measure_state_sv, filter_unmeasured=True)

        # 打印统计结果
        self._print_final_counts()
        self._finalize_binary_files()
        print("\n\n          !SUCCESS!")
        print(" XQI: Quantum Computing Program is Terminated Normally!  \n")


    def _prepare_final_state_files(self):
        """【开头】创建文件，写入对齐的文件头"""
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()
        self.state_dat_path = os.path.join(log_dir, "XQI-QC-state.dat")
        self.dm_state_dat_path = os.path.join(log_dir, "XQI-QC-Density-Matrix-state.dat")

        # 1. 状态矢量文件初始化
        with open(self.state_dat_path, "wb") as f:
            # Tag: 'XQI-QC' (6 bytes) + padding (2 bytes) = 8 bytes
            f.write(b'XQI-QC'.ljust(8, b'\x00'))
            # <I 表示小端序 unsigned int (4字节)
            f.write(struct.pack('<I', self.env.qreg_size))
            f.write(struct.pack('<I', self.shot_total))

        # 2. 密度矩阵文件初始化
        with open(self.dm_state_dat_path, "wb") as f:
            # Tag: 21 bytes + padding (3 bytes) = 24 bytes
            tag = b'XQI-QC-Density-Matrix'
            f.write(tag.ljust(24, b'\x00'))
            f.write(struct.pack('<I', self.env.qreg_size))
            f.write(struct.pack('<I', self.shot_total))

    def _append_shot_states_to_binary(self):
        """每轮 Shot 追加数据，确保使用小端序 float32"""
        # 写入状态矢量 (Before_measure_state)
        with open(self.state_dat_path, "ab") as f:
            for val in self.env.state_vector:
                # '<ff' 强制小端序，2个 float32 (real, imag)
                f.write(struct.pack('<ff', float(val.real), float(val.imag)))

        # 写入密度矩阵
        with open(self.dm_state_dat_path, "ab") as f:
            dim = self.env.density_matrix.shape[0]
            for i in range(dim):
                for j in range(dim):
                    val = self.env.density_matrix[i, j]
                    f.write(struct.pack('<ff', float(val.real), float(val.imag)))

    def _finalize_binary_files(self):
        """末尾写入对齐的 COUNT 统计信息"""
        # 写入状态矢量统计
        with open(self.state_dat_path, "ab") as f:
            # 'COUNT' (5 bytes) + padding (3 bytes) = 8 bytes
            f.write(b'COUNT'.ljust(8, b'\x00'))
            for count in self.shots_count_sv:
                f.write(struct.pack('<I', int(count)))

        # 写入密度矩阵统计
        with open(self.dm_state_dat_path, "ab") as f:
            f.write(b'COUNT'.ljust(8, b'\x00'))
            for count in self.shots_count_dm:
                f.write(struct.pack('<I', int(count)))

    def execute_instruction(self, node):
        opcode = next(c.value for c in node.children if c.type == 'Opcode')

        # 分发到各执行函数
        method_name = f"execute_{opcode.lower().replace('-', '_')}"
        if hasattr(self, method_name):
            getattr(self, method_name)(node)
        else:
            self.execute_branch(node, opcode)

        # 全局噪声应用逻辑
        # 仅在量子门之后且当前环境开启了 error_model 时应用
        if opcode in ['U3', 'CNOT', 'GPS'] and self.env.error_model:
            # 检查下一条指令是不是 ERR。如果是，则跳过全局噪声，改用 ERR 提供的局部噪声
            has_local_err = False
            if self.env.pc + 1 < len(self.body_instructions):
                next_instr = self.body_instructions[self.env.pc + 1]
                next_opcode = next(c.value for c in next_instr.children if c.type == 'Opcode')
                if next_opcode == 'ERR':
                    has_local_err = True

            if not has_local_err:
                qubits = self._get_affected_qubits(node)
                self.env.apply_quantum_noise(qubits)

    def execute_err(self, node):
        """
        执行局部误差指令：ERR(model, code, p1, p2, p3) q[n];
        """
        operands_node = next(c for c in node.children if c.type == 'Operands')
        ops = operands_node.children

        # 解析参数 (按照 Parser 定义的展平结构)
        # 格式：[model, code, p1, p2, p3, ..., qreg1, qreg2...]
        model_type = int(ops[0].value)
        # 找到量子寄存器的起始位置
        q_start_idx = 0
        for i, op in enumerate(ops):
            if op.value.startswith('q['):
                q_start_idx = i
                break

        # 提取物理参数
        params = [float(op.value) for op in ops[2:q_start_idx]]
        # 提取目标量子位
        target_qubits = [self._parse_register_index(op.value, 'q') for op in ops[q_start_idx:]]

        # 应用局部噪声
        self.env.convert_to_density()
        for qubit in target_qubits:
            # 调用 Environment 中已经实现的 Kraus 生成逻辑
            kraus_ops = self.env.generate_kraus_operators(model_type, qubit, params)
            self.env.apply_kraus_channel(kraus_ops)

    def execute_shot(self, node):
        pass

    def execute_error(self, instruction_node):
        # 1. 提取 Operands 子节点
        operands_node = next((c for c in instruction_node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise XQIExecutionError("error instruction requires at least the enable parameter")

        operands = operands_node.children

        # 2. 处理第一个参数：Enable/Disable
        enable_str = operands[0].value
        if enable_str in ['TRUE', '1']:
            enable = True
        elif enable_str in ['FALSE', '0']:
            enable = False
        else:
            raise XQIExecutionError("First parameter must be TRUE/FALSE or 1/0")

        # 如果关闭错误模型，直接返回
        if not enable:
            self.env.error_model = None
            if hasattr(self.env, '_pending_error_model'):
                del self.env._pending_error_model
            return

        # 3. 处理第二个参数：Error Code
        code = config.DEFAULT_Q_ERROR_CODE if len(operands) < 2 else int(operands[1].value)

        # 4. 根据 Code 解析后续参数
        if code == 1:
            # --- 去极化误差逻辑 ---
            # 参数顺序: enable, code, p1, p2, p_measure, p_reset
            p1 = float(operands[2].value) if len(operands) > 2 else config.DEFAULT_Q1_ERROR_PROBABILITY
            p2 = float(operands[3].value) if len(operands) > 3 else config.DEFAULT_Q2_ERROR_PROBABILITY
            p_measure = float(operands[4].value) if len(operands) > 4 else config.DEFAULT_MEASURE_ERROR_PROBABILITY
            p_reset = float(operands[5].value) if len(operands) > 5 else config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, p1, p2, p_measure, p_reset)

        elif code in [2, 3]:
            # --- 幅度(2)或相位(3)衰减误差逻辑 ---
            # 参数顺序: enable, code, gamma, p_measure, p_reset
            # 根据 code 选择 config 中的默认 gamma
            default_gamma = config.DEFAULT_AMP_DAMPING_GAMMA if code == 2 else config.DEFAULT_PHASE_DAMPING_GAMMA

            # 第三个参数是 gamma
            gamma = float(operands[2].value) if len(operands) > 2 else default_gamma
            # 第四个参数是测量误差 (原本是第五个)
            p_measure = float(operands[3].value) if len(operands) > 3 else config.DEFAULT_MEASURE_ERROR_PROBABILITY
            # 第五个参数是重置误差 (原本是第六个)
            p_reset = float(operands[4].value) if len(operands) > 4 else config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, gamma, gamma, p_measure, p_reset)
        elif code == 4:  # Thermal Relaxation
            # 指令格式: error TRUE, 4, T1, T2, Tgate, p_measure, p_reset
            t1 = float(operands[2].value) if len(operands) > 2 else config.DEFAULT_THERMAL_RELAXATION_ERROR_T1
            t2 = float(operands[3].value) if len(operands) > 3 else config.DEFAULT_THERMAL_RELAXATION_ERROR_T2
            tg = float(operands[4].value) if len(operands) > 4 else config.DEFAULT_THERMAL_RELAXATION_ERROR_TGATE
            p_measure = float(operands[5].value) if len(operands) > 5 else config.DEFAULT_MEASURE_ERROR_PROBABILITY
            p_reset = float(operands[6].value) if len(operands) > 6 else config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, [t1, t2, tg], [t1, t2, tg], p_measure, p_reset)
        elif code == 5:  # Pauli Error
            # 指令格式: error TRUE, 5, px, py, pz, p_measure, p_reset
            px = float(operands[2].value) if len(operands) > 2 else config.DEFAULT_PAULI_X_ERROR_PROBABILITY
            py = float(operands[3].value) if len(operands) > 3 else config.DEFAULT_PAULI_Y_ERROR_PROBABILITY
            pz = float(operands[4].value) if len(operands) > 4 else config.DEFAULT_PAULI_Z_ERROR_PROBABILITY
            p_measure = float(operands[5].value) if len(operands) > 5 else config.DEFAULT_MEASURE_ERROR_PROBABILITY
            p_reset = float(operands[6].value) if len(operands) > 6 else config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, [px, py, pz], [px, py, pz], p_measure, p_reset)
        elif code == 6:  # Coherent Error
            # 指令格式: error TRUE, 6, ex, ey, ez, p_measure, p_reset
            ex = float(operands[2].value) if len(operands) > 2 else config.DEFAULT_COHERENT_X_UNITARY_ERROR_PROBABILITY
            ey = float(operands[3].value) if len(operands) > 3 else config.DEFAULT_COHERENT_Y_UNITARY_ERROR_PROBABILITY
            ez = float(operands[4].value) if len(operands) > 4 else config.DEFAULT_COHERENT_Z_UNITARY_ERROR_PROBABILITY
            p_measure = float(operands[5].value) if len(operands) > 5 else config.DEFAULT_MEASURE_ERROR_PROBABILITY
            p_reset = float(operands[6].value) if len(operands) > 6 else config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, [ex, ey, ez], [ex, ey, ez], p_measure, p_reset)

        else:
            # 其他 Code 逻辑（暂按默认处理）
            p1 = config.DEFAULT_Q1_ERROR_PROBABILITY
            p2 = config.DEFAULT_Q2_ERROR_PROBABILITY
            p_measure = config.DEFAULT_MEASURE_ERROR_PROBABILITY
            p_reset = config.DEFAULT_RESET_ERROR_PROBABILITY
            self.env.error_model = (code, p1, p2, p_measure, p_reset)

        # 处理延迟应用逻辑
        if hasattr(self.env, '_pending_error_model'):
            self.env._pending_error_model = self.env.error_model
        # print(f"Error model updated: code={code}, p1={p1}, p2={p2}, p_m={p_measure}, p_r={p_reset}")
    def execute_qreg(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        operand_str = operands_node.children[0].value
        left, right = operand_str.find('['), operand_str.find(']')
        qreg_size = int(operand_str[left + 1:right])
        self.env.resize_qreg(qreg_size)
        print(f"Quantum Register Number: {qreg_size}")
    def execute_creg(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        operand_str = operands_node.children[0].value
        left, right = operand_str.find('['), operand_str.find(']')
        creg_size = int(operand_str[left + 1:right])
        self.env._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.env.creg = self.env._initial_creg.copy()
        # 匹配 C 语言 613 行
        print(f"Classical Register Number: {creg_size}")
    def execute_mov(self, node):
        """
        执行 MOV 指令，支持：
        - 普通寄存器间 MOV
        - 立即数到寄存器
        - PC/LR ↔ 寄存器/立即数
        - 寄存器/立即数 → SF / ZF（会直接影响标志位）
        - 写入普通 R 寄存器后自动更新 SF 和 ZF
        """
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise XQIExecutionError("MOV instruction requires 2 operands")
        dest_str = operands_node.children[0].value.strip()
        src_str = operands_node.children[1].value.strip()
        # 解析目标
        if dest_str in {'PC', 'LR', 'SF', 'ZF'}:
            dest_type = dest_str
            dest_val = None
        elif dest_str.startswith('R[') and dest_str.endswith(']'):
            dest_type = 'R'
            try:
                dest_val = int(dest_str[2:-1])
            except ValueError:
                raise XQIExecutionError(f"Invalid R register format: {dest_str}") from None
            if dest_val < 0 or dest_val >= len(self.env.registers):
                raise XQIExecutionError(f"R[{dest_val}] out of range")
        else:
            raise XQIExecutionError(f"Unsupported destination: {dest_str}")
        # 解析源（支持立即数、R寄存器、PC、LR）
        if src_str in {'PC', 'LR', 'SF', 'ZF'}:
            src_type = src_str
            src_val = None
        elif src_str.startswith('R[') and src_str.endswith(']'):
            src_type = 'R'
            try:
                src_val = int(src_str[2:-1])
            except ValueError:
                raise XQIExecutionError(f"Invalid source R register: {src_str}") from None
            if src_val < 0 or src_val >= len(self.env.registers):
                raise XQIExecutionError(f"Source R[{src_val}] out of range")
        elif src_str.isdigit() or (src_str.lstrip('-').isdigit()):
            src_type = 'imm'
            src_val = int(src_str) # 优先尝试整数
        elif '.' in src_str or 'e' in src_str.lower() or 'E' in src_str:
            src_type = 'imm'
            src_val = float(src_str) # 浮点数
        else:
            raise XQIExecutionError(f"Unsupported source operand: {src_str}")
        # 执行 MOV
        if dest_type == 'R':
            # 普通寄存器目标
            if src_type == 'R':
                value = self.env.registers[src_val]
            elif src_type == 'imm':
                value = src_val
            elif src_type == 'PC':
                value = self.env.pc
            elif src_type == 'LR':
                value = self.env.lr
            elif src_type == 'SF':
                value = float(self.env.SF)
            elif src_type == 'ZF':
                value = float(self.env.ZF)
            else:
                raise XQIExecutionError(f"Unsupported src → R dest: {src_type}")
            self.env.registers[dest_val] = value
            # 写入普通 R 寄存器后，更新标志位（模仿算术指令）
            self._set_flags(value)
        elif dest_type == 'PC':
            old_pc = self.env.pc
            if src_type == 'imm' and src_val == 0:
                # 特殊语义：结束当前 shot
                self.env.shot_completed = True
                # print("MOV PC,0 detected → marking current shot as completed")
                return
            else:
                # 跳转到 PC
                if src_type == 'R':
                    self.env.pc = int(self.env.registers[src_val])
                elif src_type == 'imm':
                    self.env.pc = int(src_val)
                elif src_type == 'LR':
                    self.env.pc = self.env.lr
                    # print(f"MOV PC, LR → jumping to {self.env.pc} (LR value is {self.env.lr})")
                elif src_type in {'SF', 'ZF'}:
                    raise XQIExecutionError("Cannot MOV SF/ZF directly to PC")
                else:
                    raise XQIExecutionError(f"Unsupported src → PC: {src_type}")
                new_pc = self.env.pc
                # print(f"PC ← {src_str} (value={new_pc}, old_pc was {old_pc}, LR={self.env.lr})")
                # PC 变更不更新标志位
        elif dest_type == 'LR':
            # 设置链接寄存器
            if src_type == 'R':
                self.env.lr = int(self.env.registers[src_val])
            elif src_type == 'imm':
                self.env.lr = int(src_val)
            elif src_type == 'PC':
                self.env.lr = self.env.pc
            elif src_type in {'SF', 'ZF'}:
                raise XQIExecutionError("Cannot MOV SF/ZF directly to LR")
            else:
                raise XQIExecutionError(f"Unsupported src → LR: {src_type}")
            # LR 变更不更新标志位
        elif dest_type == 'SF':
            # 直接设置符号标志
            if src_type == 'imm':
                val = int(src_val)
            elif src_type == 'R':
                val = int(self.env.registers[src_val])
            elif src_type in {'PC', 'LR'}:
                raise XQIExecutionError("Cannot MOV PC/LR to SF")
            else:
                val = int(src_val) # SF/ZF
            self.env.SF = 1 if val != 0 else 0 # 通常 SF=1 表示负数，这里简化处理为非零即1
        elif dest_type == 'ZF':
            # 直接设置零标志
            if src_type == 'imm':
                val = int(src_val)
            elif src_type == 'R':
                val = int(self.env.registers[src_val])
            elif src_type in {'PC', 'LR'}:
                raise XQIExecutionError("Cannot MOV PC/LR to ZF")
            else:
                val = int(src_val)
            self.env.ZF = 1 if val == 0 else 0
        else:
            raise XQIExecutionError(f"Unsupported destination type: {dest_type}")
        # 调试输出
        # print(f"MOV {dest_str} <- {src_str} completed.")
    def execute_u3(self, node):
        # 1. 解析参数
        theta = self._parse_parameter(node.children[1].children[0], 'R')
        phi = self._parse_parameter(node.children[1].children[1], 'R')
        lam = self._parse_parameter(node.children[1].children[2], 'R')
        qubit = self._parse_register_index(node.children[1].children[3].value, 'q')
        # 2. 构造 U3 矩阵 (严格匹配 C 语言公式)
        u_gate = np.array([
            [np.cos(theta / 2), -1j * np.exp(1j * lam) * np.sin(theta / 2)],
            [-1j * np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
        ], dtype=np.complex128)
        # 3. 应用门
        full_u = self.env.get_full_operator(u_gate, qubit)
        self.env.apply_unitary(full_u)

    def execute_cnot(self, node):
        control = self._parse_register_index(node.children[1].children[0].value, 'q')
        target = self._parse_register_index(node.children[1].children[1].value, 'q')
        # 构造全系统 CNOT: |0><0|⊗I + |1><1|⊗X
        P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        # 构造两个分支的张量积并求和
        op0 = np.array([[1.0]], dtype=np.complex128)
        op1 = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.env.qreg_size - 1, -1, -1):
            # 分支 0
            gate0 = P0 if i == control else I
            op0 = np.kron(op0, gate0)
            # 分支 1
            gate1 = P1 if i == control else (X if i == target else I)
            op1 = np.kron(op1, gate1)
        full_cnot = op0 + op1
        self.env.apply_unitary(full_cnot)

    def execute_gps(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise XQIExecutionError("GPS requires delta and qubit parameters")
        # 解析delta参数（支持立即数或R寄存器）
        delta = self._parse_parameter(operands_node.children[0], 'R')
        # 验证第二个参数格式（q寄存器）
        self._parse_register_index(operands_node.children[1].value, 'q')
        # 应用全局相位
        phase = np.exp(1j * delta)
        if self.env.simulation_mode == 'statevector':
            # 注意:quantum_state 恒为 density_matrix 的别名,态矢量必须用 state_vector
            self.env.state_vector = self.env.state_vector * phase
        else:
            # 全局相位对密度矩阵无可观测效应:e^{iδ} ρ e^{-iδ} = ρ,物理上恒等,无需计算
            pass

    def execute_measure(self, node):
        operands = next(c for c in node.children if c.type == 'Operands').children
        qubit = self._parse_register_index(operands[0].value, 'q')
        creg_idx = self._parse_register_index(operands[1].value, 'c')
        dim = 2 ** self.env.qreg_size

        # --- 1. State Vector (理想采样) ---
        probs_sv = [0.0, 0.0]
        for i in range(dim):
            bit = (i >> qubit) & 1
            probs_sv[bit] += np.abs(self.env.state_vector[i]) ** 2

        p_sum = np.sum(probs_sv)
        outcome_sv = self.env.rng.choice([0, 1], p=probs_sv / (p_sum if p_sum > 0 else 1))

        # 理想态坍缩
        new_sv = np.zeros_like(self.env.state_vector)
        for i in range(dim):
            if ((i >> qubit) & 1) == outcome_sv:
                new_sv[i] = self.env.state_vector[i]
        norm = np.linalg.norm(new_sv)
        self.env.state_vector = new_sv / norm if norm > 0 else new_sv

        # 记录理想结果用于 SV COUNT
        if self.env.creg_ideal is None: self.env.creg_ideal = np.zeros_like(self.env.creg)
        self.env.creg_ideal[creg_idx] = outcome_sv

        # --- 2. Density Matrix (物理采样) ---
        proj0 = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            if ((i >> qubit) & 1) == 0: proj0[i, i] = 1.0

        prob0_dm = np.real(np.trace(proj0 @ self.env.density_matrix))

        outcome_dm = 0 if self.env.rng.random() < prob0_dm else 1

        # 物理态坍缩
        proj_dm = proj0 if outcome_dm == 0 else (np.eye(dim) - proj0)
        self.env.density_matrix = proj_dm @ self.env.density_matrix @ proj_dm.conj().T
        trace = np.real(np.trace(self.env.density_matrix))
        self.env.density_matrix = self.env.density_matrix / trace if trace > 1e-15 else self.env.density_matrix
        self.env.quantum_state = self.env.density_matrix

        # 记录物理结果用于 DM COUNT
        self.env.creg[creg_idx] = outcome_dm

        # 标记哪些位被测量过 (匹配 C 语言 Measure_Event_Quantum_Register_Bit)
        self.env.measured_bits.add(qubit)
    def execute_add(self, node):
        self._execute_binary_arithmetic(node, 'add')
    def execute_sub(self, node):
        self._execute_binary_arithmetic(node, 'sub')
    def execute_mul(self, node):
        self._execute_binary_arithmetic(node, 'mul')
    def execute_div(self, node):
        self._execute_binary_arithmetic(node, 'div')

    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _get_affected_qubits(instr_node):
        """从 Instruction 节点提取受影响的量子位"""
        # 先找到 Opcode 子节点
        opcode_node = next((c for c in instr_node.children if c.type == 'Opcode'), None)
        if not opcode_node:
            return []
        opcode = opcode_node.value
        # 找到 Operands 子节点
        operands_node = next((c for c in instr_node.children if c.type == 'Operands'), None)
        if not operands_node:
            return []
        children = operands_node.children
        if opcode == 'CNOT':
            if len(children) < 2:
                return []
            # 假设格式 q[control], q[target]
            control_str = children[0].value
            target_str = children[1].value
            try:
                control = int(control_str[2:-1])
                target = int(target_str[2:-1])
                return [control, target]
            except (ValueError, IndexError):
                return []
        elif opcode in ['U3', 'measure', 'reset']:
            # 最后一个操作数是量子位
            if not children:
                return []
            qubit_str = children[-1].value
            try:
                qubit = int(qubit_str[2:-1])
                return [qubit]
            except (ValueError, IndexError):
                return []
        return []

    def _parse_parameter(self, operand_node, register_prefix):
        """解析混合参数（立即数/寄存器）"""
        value_str = operand_node.value
        try:
            # 尝试解析为立即数
            return float(value_str)
        except ValueError:
            # 解析为寄存器值
            reg_index = self._parse_register_index(value_str, register_prefix)
            return self.env.registers[reg_index]

    # 通用寄存器解析方法
    def _parse_register_index(self, reg_str, prefix):
        # 对偏移量的支持
        if '+' in reg_str:
            base_part, offset_part = reg_str.split('+', 1)
            base = self._parse_single_register(base_part, prefix)
            offset = int(offset_part.strip(' ]'))
            return base + offset
        return self._parse_single_register(reg_str, prefix)
    def _parse_single_register(self, reg_str, prefix):
        # 带详细错误信息的解析
        if not reg_str.startswith(prefix):
            raise XQIExecutionError(f"Expected {prefix} register, got {reg_str}")
        try:
            index = int(reg_str[len(prefix) + 1:-1]) # 解析类似 q[5] 的格式
        except (ValueError, IndexError):
            raise XQIExecutionError(f"Invalid register format: {reg_str}")
        # 寄存器范围检查
        max_size = self.env.qreg_size if prefix == 'q' else \
            len(self.env.registers) if prefix == 'R' else \
                len(self.env.creg)
        if index >= max_size:
            raise XQIExecutionError(f"{prefix} register index out of range: {index} (max={max_size - 1})")
        return index
    def _execute_binary_arithmetic(self, node, operation):
        operands = node.children[1].children
        # print(f"[{operation.upper()}] raw operands: {[op.value for op in operands]}")
        if len(operands) < 3:
            raise XQIExecutionError(f"{operation.upper()} needs 3 operands")
        dest_idx = self._parse_register_index(operands[0].value, 'R')
        src1_parsed = self._parse_operand(operands[1].value)
        src2_parsed = self._parse_operand(operands[2].value)
        # print(f"dest: {dest_idx}, src1_parsed: {src1_parsed}, src2_parsed: {src2_parsed}")
        if isinstance(src1_parsed, int) and operands[1].value.startswith('R['):
            val1 = self.env.registers[src1_parsed]
        else:
            val1 = float(src1_parsed) # 立即数
        if isinstance(src2_parsed, int) and operands[2].value.startswith('R['):
            val2 = self.env.registers[src2_parsed]
        else:
            val2 = float(src2_parsed)
        # print(f"values → val1={val1} (from R{src1_parsed}), val2={val2} (from R{src2_parsed})")
        if operation == 'add':
            result = val1 + val2
        elif operation == 'sub':
            result = val1 - val2
        elif operation == 'mul':
            result = val1 * val2
        elif operation == 'div':
            if abs(val2) < 1e-12:
                raise XQIExecutionError("Division by zero")
            result = val1 / val2
        else:
            raise XQIExecutionError(f"Unknown operation: {operation}")
        self.env.registers[dest_idx] = result
        self._set_flags(result)
    def _parse_operand(self, operand_str):
        operand_str = operand_str.strip()
        if operand_str.startswith('R['):
            return self._parse_register_index(operand_str, 'R')
        elif operand_str.startswith('q['):
            return self._parse_register_index(operand_str, 'q')
        elif operand_str.startswith('c['):
            return self._parse_register_index(operand_str, 'c')
        else:
            # 尝试数字
            try:
                if '.' in operand_str or 'e' in operand_str.lower():
                    return float(operand_str)
                else:
                    return int(operand_str)
            except ValueError:
                raise XQIExecutionError(f"Cannot parse operand: {operand_str}")
    # 通用标志位设置方法
    def _set_flags(self, value):
        self.env.SF = 1 if value < 0 else 0
        self.env.ZF = 1 if abs(value) < 1e-10 else 0
    def execute_branch(self, node, opcode):
        label = self._parse_label_operand(node)
        if opcode == 'B':
            self.env.pc = self._get_label_address(label)
            return
        if opcode == 'BL':
            self.env.lr = self.env.pc + 1 # 注意：这里 +1 是返回后下一条指令的地址
            self.env.pc = self._get_label_address(label)
            return
        if opcode == 'BNE':
            if self._condition_met('NE'):
                # print(f" → branch taken to {label} @ {self.labels[label]}")
                pass
            else:
                # print(" → branch NOT taken (ZF==1)")
                pass
        # 条件分支
        if self._condition_met(opcode[1:]): # BEQ → 'EQ', BNE → 'NE' 等
            self.env.pc = self._get_label_address(label)
        # 不跳转时 → 外层 +1

    @staticmethod
    def _require_operands(node, opcode, count, detail):
        """取指令的 Operands 子节点并校验操作数数量,统一缺失时的报错形式。"""
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < count:
            raise XQIExecutionError(f"{opcode} requires {count} operands ({detail})")
        return operands_node.children

    def execute_ldr(self, node):
        ops = self._require_operands(node, "LDR", 2, "dest_reg, src_mem")
        dest_reg = self._parse_register_index(ops[0].value, 'R')
        src_mem = self._parse_memory_address(ops[1].value)
        self.env.registers[dest_reg] = self.env.memory[src_mem]

    def execute_str(self, node):
        ops = self._require_operands(node, "STR", 2, "src_reg, dest_mem")
        src_reg = self._parse_register_index(ops[0].value, 'R')
        dest_mem = self._parse_memory_address(ops[1].value)
        self.env.memory[dest_mem] = self.env.registers[src_reg]

    def execute_cldr(self, node):
        ops = self._require_operands(node, "CLDR", 3, "dest_creg, real_mem, imag_mem")
        dest_creg = self._parse_register_index(ops[0].value, 'c')
        real_mem = self._parse_memory_address(ops[1].value)
        imag_mem = self._parse_memory_address(ops[2].value)
        self.env.creg[dest_creg] = complex(self.env.memory[real_mem], self.env.memory[imag_mem])

    def execute_cstr(self, node):
        ops = self._require_operands(node, "CSTR", 3, "src_creg, real_mem, imag_mem")
        src_creg = self._parse_register_index(ops[0].value, 'c')
        real_mem = self._parse_memory_address(ops[1].value)
        imag_mem = self._parse_memory_address(ops[2].value)
        complex_val = self.env.creg[src_creg]
        self.env.memory[real_mem] = complex_val.real
        self.env.memory[imag_mem] = complex_val.imag
    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _parse_label_operand(node):
        """从分支指令节点解析标签操作数"""
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise XQIExecutionError(f"{node.children[0].value} instruction missing label operand")
        label_operand = operands_node.children[0]
        if label_operand.type != 'Label':
            raise XQIExecutionError(f"Expected label operand, got {label_operand.type}")
        return label_operand.value.strip(':') # 去除可能存在的冒号
    def _get_label_address(self, label):
        label = label.strip(':')
        if label not in self.labels:
            raise XQIExecutionError(f"Undefined label: {label}")
        return self.labels[label]
    # 条件检查方法 -----------------------------------------------
    def _condition_met(self, condition_type):
        """通用条件检查方法"""
        sf, zf = self.env.SF, self.env.ZF
        conditions = {
            'EQ': lambda: zf == 1,
            'NE': lambda: zf == 0,
            'GT': lambda: sf == 0 and zf == 0,
            'GE': lambda: sf == 0 or zf == 1,
            'LT': lambda: sf == 1 and zf == 0,
            'LE': lambda: sf == 1 or zf == 1
        }
        return conditions.get(condition_type, lambda: False)()

    def execute_cmp(self, node):
        """CMP a, b:计算 (a - b) 并置 SF/ZF(不保存结果)。

        标志位约定与 _set_flags 及 6 个条件分支保持一致:
        a == b → ZF=1(EQ);a < b(有符号)→ SF=1(LT);a > b → SF=0 且 ZF=0(GT)。
        操作数支持立即数或 R 寄存器。
        """
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise XQIExecutionError("CMP requires 2 operands")
        val1 = self._parse_parameter(operands_node.children[0], 'R')
        val2 = self._parse_parameter(operands_node.children[1], 'R')
        self._set_flags(val1 - val2)

    def execute_bx(self, node):
        """BX LR:跳转到 LR 保存的返回地址(等价 MOV PC,LR 的返回惯用法)。

        仅支持 LR 操作数;其他寄存器/标签操作数报错。
        """
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise XQIExecutionError("BX requires an operand (only LR is supported)")
        target = operands_node.children[0].value.strip()
        if target != 'LR':
            raise XQIExecutionError(f"BX only supports LR, got {target}")
        self.env.pc = int(self.env.lr)

    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _parse_memory_address(addr_str):
        """解析内存地址格式 M[数字] 或 M[基址+偏移]"""
        if not addr_str.startswith('M'):
            raise XQIExecutionError(f"Invalid memory address format: {addr_str}")
        # 提取方括号内容
        left = addr_str.find('[')
        right = addr_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise XQIExecutionError(f"Invalid memory address format: {addr_str}")
        # 解析基址和偏移量
        expr = addr_str[left + 1:right]
        if '+' in expr:
            base, offset = expr.split('+', 1)
            return int(base) + int(offset)
        else:
            return int(expr)
    def execute_debug(self, node):
        self.print_debug_info(self.shot_idx)
    def execute_debug_p(self, node):
        self.print_debug_info(self.shot_idx)
        input("Press 'p' to continue: ")
    def execute_reset(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        qubit = self._parse_register_index(operands_node.children[0].value, 'q')
        # execute_error 只产生 code 1-6,其中不含 reset 概率项,物理重置恒为理想重置
        self.env.apply_physical_reset(qubit, 0.0)
    def execute_barrier(self, node):
        # No-op in this simulation
        pass
    def execute_rand(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise XQIExecutionError("rand requires 2 operands (dest_reg, seed_reg)")
        # 操作数在 Operands 子节点中;children[0] 是 Opcode 节点,不可按其取寄存器号
        dest = self._parse_register_index(operands_node.children[0].value, 'R')
        seed = self._parse_register_index(operands_node.children[1].value, 'R')
        # 用种子寄存器构造独立的局部随机源,不扰动测量使用的实例随机流
        self.env.registers[dest] = np.random.default_rng(int(self.env.registers[seed])).uniform(0, 1)

    def print_debug_info(self, shot_id=1):
        if not self._debug_file_initialized:
            self._initialize_debug_files()

        q_size = self.env.qreg_size
        dim = 2 ** q_size

        # 直接获取两种状态，不再进行模式转换判断
        psi = self.env.state_vector
        rho = self.env.density_matrix

        # --- 1. 写入 State Vector 文本 (XQI-QC-list.txt) ---
        with open(self.paths['sv_txt'], "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")
            f.write("XQI: current states (High Qubit->Low Qubit) :\n")
            for i in range(dim):
                bin_str = format(i, f'0{q_size}b')
                f.write(f"state  {bin_str:>10s}:  ({psi[i].real:.6f})+({psi[i].imag:.6f})i\n")

            f.write("corresponding Density Matrix state:\n")
            # 状态矢量对应的密度矩阵是其自身的外积
            self._write_matrix_to_text(f, np.outer(psi, psi.conj()))

            f.write("\ncurrent states probability (High Qubit->Low Qubit) :\n")
            for i in range(dim):
                bin_str = format(i, f'0{q_size}b')
                f.write(f"state {bin_str:>10s}:  {np.abs(psi[i]) ** 2:.6f}\n")
            self._write_common_debug(f)

        # --- 2. 写入 Density Matrix 文本 (XQI-QC-Density-Matrix-list.txt) ---
        with open(self.paths['dm_txt'], "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")
            f.write("XQI: current Density Matrix states:\n")
            self._write_matrix_to_text(f, rho)
            self._write_common_debug(f)

        # --- 3. 写入二进制数据 (.dat) ---
        # 显式使用 float32 确保与 C 语言 float 兼容;
        # 统一 '<ff' 显式小端,与 state .dat 的写入格式保持一致
        with open(self.paths['sv_dat'], "ab") as f:
            for val in psi:
                f.write(struct.pack('<ff', float(val.real), float(val.imag)))

        with open(self.paths['dm_dat'], "ab") as f:
            for i in range(dim):
                for j in range(dim):
                    val = rho[i, j]
                    f.write(struct.pack('<ff', float(val.real), float(val.imag)))
    def _initialize_debug_files(self):
        """
        初始化所有调试文件。
        如果是本次运行的第一次 debug，则删除旧文件并写入 Header。
        """
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()
        # 统一定义文件路径
        self.paths = {
            'sv_txt': os.path.join(log_dir, config.FILENAME_DEBUG),
            'sv_dat': os.path.join(log_dir, config.FILENAME_DEBUG_MATLAB),
            'dm_txt': os.path.join(log_dir, config.FILENAME_DEBUG_DENSITY_MATRIX),
            'dm_dat': os.path.join(log_dir, config.FILENAME_DEBUG_DENSITY_MATRIX_MATLAB)
        }
        # --- 第一步：物理删除已存在的旧文件 ---
        for path in self.paths.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    # print(f"Debug file cleaned: {os.path.basename(path)}")
                except OSError as e:
                    print(f"Error cleaning debug file {path}: {e}")
        # --- 第二步：初始化文本文件 (Header) ---
        header_text = self.source_code_text.strip() + "\n\n"
        header_text += "Label Number Sequence Label Symbol\n"
        for idx, (seq, symbol) in enumerate(self.labels_info):
            header_text += f"Label {idx:3d}: {seq:3d} {symbol}\n"
        header_text += "\n\n"
        with open(self.paths['sv_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)
        with open(self.paths['dm_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)
        # --- 第三步：初始化二进制文件 (Header) ---
        # 1. SV Binary Header: Tag(6 bytes) + QregSize(4 bytes)
        with open(self.paths['sv_dat'], "wb") as f:
            f.write(b'XQI-QC') # 对应 C 语言 char[6]
            f.write(struct.pack('I', self.env.qreg_size)) # 对应 C 语言 unsigned int
        # 2. DM Binary Header: Tag(21 bytes) + QregSize(4 bytes)
        with open(self.paths['dm_dat'], "wb") as f:
            tag_dm = b'XQI-QC-Density-Matrix' # 21字节
            f.write(tag_dm)
            f.write(struct.pack('I', self.env.qreg_size))
        # 标记初始化完成
        self._debug_file_initialized = True
    @staticmethod
    def _write_matrix_to_text(f, matrix):
        rows, cols = matrix.shape
        f.write(f"\nmatrix rows:{rows}, matrix columns:{cols}:\n")
        for i in range(rows):
            for j in range(cols):
                val = matrix[i, j]
                f.write(f"[{i}][{j}]:({val.real:.6f})+({val.imag:.6f})i\n")
        f.write("\n")
    def _write_common_debug(self, f):
        """写入寄存器、CPSR 和内存的通用部分"""
        f.write("\n register:\n")
        for idx, val in enumerate(self.env.registers):
            f.write(f"R[{idx:2d}]={val:15.10f}\n")
        f.write("\nCPSR: ")
        f.write(f"SIGN_FLAG={self.env.SF}; " if self.env.SF else "SIGN_FLAG=0; ")
        f.write(f"ZERO_FLAG={self.env.ZF}.\n" if self.env.ZF else "ZERO_FLAG=0.\n")
        f.write("\n memory:\n")
        for idx, val in enumerate(self.env.memory):
            f.write(f"M[{idx:4d}]={val:15.10f}\n")
        f.write("\n\n")

    @staticmethod
    def _calculate_state_code(creg_array):
        """通用代码计算：将经典寄存器数组转为整数索引"""
        code = 0
        for i, val in enumerate(creg_array):
            bit = 1 if abs(val.real) > 0.5 else 0
            code += (bit << i)
        return code

    def _print_complete_measure_info(self, state_vec, filter_unmeasured=False):
        print("\nXQI Success: Complete Measure Event Information:\n")
        dim = len(state_vec)
        q_size = self.env.qreg_size
        measured_bits = getattr(self.env, 'measured_bits', set())

        for i in range(dim):
            bin_str = format(i, f'0{q_size}b')
            prob = np.abs(state_vec[i]) ** 2

            # C 语言逻辑：对于没被测量到的位，如果状态在该位上是 '1'，则该路径概率为 0
            if filter_unmeasured:
                for q_idx in range(q_size):
                    if q_idx not in measured_bits:
                        # 匹配 C 语言：Transform_Unsigned_Decimal_to_Binary 的索引
                        if bin_str[q_size - 1 - q_idx] == '1':
                            prob = 0.0
                            break

            print(f"state:        {bin_str}: probability={prob:.6f}")
            print(f"resultant measure state:")
            print(f"matrix rows:{dim}, matrix columns:1:")
            for j in range(dim):
                # 只有对应基矢的分量保留，其他为 0 (模拟投影)
                val = state_vec[j] if (i == j and prob > 0) else 0.0j
                print(f"[{j}][0]:({val.real:.6f})+({val.imag:.6f})i")
            print("")

    def _print_final_counts(self):
        q_size = self.env.qreg_size
        num_states = 2 ** q_size

        print("\nXQI Success: After measure, STATE COUNT:")
        print(f"Total Count:   {self.shot_total}")
        for i in range(num_states):
            bin_str = format(i, f'0{q_size}b')
            count = self.shots_count_sv[i]
            print(f"State:        {bin_str}:     Count={count:6d},    Probability={count / self.shot_total:.6f}")

        print("\n\nXQI Success: After measure, STATE COUNT (Density Matrix):")
        print(f"Total Count:   {self.shot_total}")
        for i in range(num_states):
            bin_str = format(i, f'0{q_size}b')
            count = self.shots_count_dm[i]
            print(f"State:        {bin_str}:     Count={count:6d},    Probability={count / self.shot_total:.6f}")