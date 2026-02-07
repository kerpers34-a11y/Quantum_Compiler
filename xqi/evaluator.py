import os
import struct
import numpy as np
from xqi import config

class ASTNode:
    def __init__(self, type, value=None, children=None, line=None, col=None):
        self.type = type
        self.value = value if value is not None else []
        self.children = children if children is not None else []
        self.line = line
        self.col = col

def collect_labels(self, ast):  # Add self
    labels = {}
    pc = 0
    for node in ast.children:
        if node.type == 'Label':
            labels[node.value] = pc
        elif node.type == 'Instruction' and node.children[0].value not in {'shot', 'qreg', 'creg', 'error', 'XQI-BEGIN', 'XQI-END'}:
            pc += 1  # Only count body instructions
    return labels

class InstructionError(ValueError):
    def __init__(self, instr_name, msg):
        super().__init__(f"[{instr_name}] {msg}")

class QuantumEnvironment:
    def __init__(self, qreg_size=0, creg_size=0, max_registers=config.MAX_Register,
                 max_memory=config.MAX_Memory, simulation_mode='statevector'):
        # 参数校验
        self.initial_quantum_state = None
        if not all(isinstance(x, int) and x >= 0 for x in [qreg_size, creg_size]):
            raise ValueError("Register sizes must be non-negative integers")
        self.qreg_size = qreg_size
        self.creg_size = creg_size
        self.simulation_mode = simulation_mode.lower()
        self.max_registers = max_registers
        self.max_memory = max_memory
        # 初始化量子寄存器大小和经典寄存器大小
        self._initial_qreg_size = qreg_size
        self._initial_creg_size = creg_size
        # 初始化量子态
        self.quantum_state = []
        self._reset_quantum_register(qreg_size)  # 调用内部重置方法
        # 初始化经典寄存器（兼容复数测量结果）
        self._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.creg = self._initial_creg.copy()
        # 初始化通用寄存器和存储
        self.registers = np.zeros(max_registers, dtype=np.float64)
        self.memory = np.zeros(max_memory, dtype=np.float64)
        # 控制寄存器
        self.pc = 0  # 程序计数器
        self.lr = 0  # 链接寄存器
        self.SF = 0  # 符号标志（负数）
        self.ZF = 0  # 零标志
        # 错误模型
        self.error_model = (config.default_Q_error_Code, config.default_Q1_error_Probability, config.default_Q2_error_Probability, config.default_measure_error_Probability, config.default_reset_error_Probability)
        self._pending_error_model = (config.default_Q_error_Code, config.default_Q1_error_Probability, config.default_Q2_error_Probability, config.default_measure_error_Probability, config.default_reset_error_Probability)

        self.initial_mode = simulation_mode.lower()  # 记录初始设定的模式
        self.simulation_mode = self.initial_mode

    def _reset_quantum_register(self, new_size):
        """重置量子寄存器"""
        self.qreg_size = new_size
        self.initial_quantum_state = self._initialize_quantum_state(new_size)
        self.quantum_state = self.initial_quantum_state.copy()

    def resize_qreg(self, new_size):
        """在 qreg 声明时调用：改变寄存器大小并初始化"""
        if not isinstance(new_size, int) or new_size < 0:
            raise ValueError("qreg_size must be non-negative integer")

        self.qreg_size = new_size
        self.initial_quantum_state = self._initialize_quantum_state(new_size)
        self.quantum_state = self.initial_quantum_state.copy()

        if hasattr(self, '_pending_error_model'):
            self.error_model = self._pending_error_model
            del self._pending_error_model

    def _initialize_quantum_state(self, size):
        dim = 2 ** size if size > 0 else 1  # Treat size=0 as dim=1
        if self.simulation_mode == 'statevector':
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0 + 0j
        elif self.simulation_mode == 'density_matrix':
            state = np.zeros((dim, dim), dtype=np.complex128)
            state[0, 0] = 1.0 + 0j
        else:
            raise ValueError(f"Unsupported simulation mode: {self.simulation_mode}")
        return state

    def reset_for_shot(self):
        """重置量子及CPSR环境到初始状态"""

        self.simulation_mode = self.initial_mode
        self.quantum_state = self.initial_quantum_state.copy()

        self.quantum_state = self.initial_quantum_state.copy()
        self.creg = self._initial_creg.copy()
        self.pc = 0
        self.lr = 0
        self.SF = 0  # 符号标志（负数）
        self.ZF = 0  # 零标志

    def full_reset(self):
        """完全重置所有状态（用于环境初始化）"""
        self.reset_for_shot()
        self.registers.fill(0.0)
        self.memory.fill(0.0)

    def convert_to_density(self):

        if not isinstance(self.quantum_state, np.ndarray):
            self.quantum_state = np.array(self.quantum_state)

        if self.quantum_state.ndim == 1:  # 只有是向量时才转换
            self.quantum_state = np.outer(self.quantum_state, self.quantum_state.conj())
        self.simulation_mode = 'density_matrix'

    def apply_quantum_noise(self, qubits):
        """应用当前配置的量子噪声"""
        if self.qreg_size == 0 or not qubits:  # Early return for trivial cases
            return
        print(f"Applying noise to qubits {qubits} with model {self.error_model}")
        if not self.error_model:
            return
        if hasattr(self, '_pending_error_model'):
            self.error_model = self._pending_error_model
            print(f"Applying delayed error model at first noise application: {self._pending_error_model}")
            del self._pending_error_model
        noise_type, p1, p2, p_measure, p_reset = self.error_model
        p = p1 if len(qubits) == 1 else p2
        kraus_ops = self.generate_kraus_operators(noise_type, qubits, [p])
        if any(not isinstance(k, np.ndarray) for k in kraus_ops):
            raise TypeError("Kraus operators must be numpy arrays")
        self.convert_to_density()
        # 应用完整噪声通道：sum K rho K†
        new_rho = np.zeros_like(self.quantum_state, dtype=np.complex128)
        for k in kraus_ops:
            if np.allclose(k, 0):  # 跳过零算子（针对p=1.0时的无效项）
                continue
            temp = k @ self.quantum_state @ k.conj().T
            new_rho += temp
        trace = np.real(np.trace(new_rho))
        if trace > 1e-12:
            new_rho /= trace
        else:
            print("Warning: trace after noise ≈ 0, resetting to maximally mixed")

            if not isinstance(self.quantum_state, np.ndarray):
                self.quantum_state = np.array(self.quantum_state)

            dim = self.quantum_state.shape[0]
            new_rho = np.eye(dim, dtype=np.complex128) / dim
        self.quantum_state = new_rho
        assert isinstance(self.quantum_state, np.ndarray), \
            f"quantum_state corrupted: {type(self.quantum_state)}"

    def generate_kraus_operators(self, noise_type, qubit_idx, params):
        p = params[0]
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        raw_ops = []
        if noise_type == 1:  # Depolarizing
            raw_ops = [(np.sqrt(1 - p), I), (np.sqrt(p / 3), X), (np.sqrt(p / 3), Y), (np.sqrt(p / 3), Z)]
        elif noise_type == 2:  # Amplitude Damping
            k0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=np.complex128)
            raw_ops = [(1.0, k0), (1.0, k1)]
        elif noise_type == 3:  # Phase Damping
            k0 = np.sqrt(1 - p) * I
            k1 = np.sqrt(p) * np.array([[1, 0], [0, 0]], dtype=np.complex128)
            k2 = np.sqrt(p) * np.array([[0, 0], [0, 1]], dtype=np.complex128)
            raw_ops = [(1.0, k0), (1.0, k1), (1.0, k2)]
        else:
            raw_ops = [(1.0, I)]  # Default Identity

        # 提升到全系统空间
        operators = []
        for coeff, m in raw_ops:
            full_op = np.eye(1, dtype=np.complex128)
            for q in reversed(range(self.qreg_size)):
                if q == qubit_idx:
                    full_op = np.kron(full_op, m)
                else:
                    full_op = np.kron(full_op, np.eye(2))
            operators.append(full_op * coeff)
        return operators

    def get_full_operator(self, op, target_qubit):
        """将单比特算符扩展到全系统空间，匹配 C 语言的高位在左原则"""
        # C 语言逻辑：q[n-1] ⊗ q[n-2] ⊗ ... ⊗ q[0]
        # 对应 Python：从最高索引开始 kron
        full_op = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.qreg_size - 1, -1, -1):  # 从高位到低位
            if i == target_qubit:
                full_op = np.kron(full_op, op)
            else:
                full_op = np.kron(full_op, np.eye(2, dtype=np.complex128))
        return full_op

    def apply_unitary(self, u_matrix):
        """执行变换：根据当前维度决定运算方式"""
        if self.quantum_state.ndim == 1:
            # 状态矢量模式：psi = U @ psi
            self.quantum_state = u_matrix @ self.quantum_state
        else:
            # 密度矩阵模式：rho = U @ rho @ U.H
            self.quantum_state = u_matrix @ self.quantum_state @ u_matrix.conj().T

    def apply_depolarizing_error(self, qubit, p):
        """严格匹配 C 语言 operation_Density_Matrix_ERROR 代码"""
        if p <= 0: return

        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # 计算四个分量: (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
        # 获取全系统空间的 Pauli 算符
        full_X = self.get_full_operator(X, qubit)
        full_Y = self.get_full_operator(Y, qubit)
        full_Z = self.get_full_operator(Z, qubit)

        rho = self.quantum_state
        self.quantum_state = (1 - p) * rho + (p / 3.0) * (
                full_X @ rho @ full_X.conj().T +
                full_Y @ rho @ full_Y.conj().T +
                full_Z @ rho @ full_Z.conj().T
        )

    def apply_physical_reset(self, target_qubit, p_reset):
        """
        严格对齐 C 语言的 operation_Density_Matrix_reset
        target_qubit: 目标比特索引
        p_reset: 重置错误率 (对应 C 语言中的 Q1_error_Probability)
        """
        self.convert_to_density()
        dim = 2 ** self.qreg_size

        # 1. 构造 C 语言中的投影/偏迹算符 (基于单比特)
        # bra0 = [1, 0], bra1 = [0, 1]
        m0 = np.array([[1, 0]], dtype=np.complex128)
        m1 = np.array([[0, 1]], dtype=np.complex128)

        # 2. 构造全系统的偏迹算符
        # 这里的逻辑是：Tr_i(rho) = M0_i rho M0_i^† + M1_i rho M1_i^†
        # 我们需要构造维度为 (dim/2, dim) 的算符
        def build_trace_op(op_single):
            full_op = np.array([[1.0]], dtype=np.complex128)
            for i in range(self.qreg_size - 1, -1, -1):
                if i == target_qubit:
                    full_op = np.kron(full_op, op_single)
                else:
                    full_op = np.kron(full_op, np.eye(2))
            return full_op

        full_m0 = build_trace_op(m0)
        full_m1 = build_trace_op(m1)

        # 计算偏迹后的状态 (维度变为 dim/2 x dim/2)
        rho_reduced = full_m0 @ self.quantum_state @ full_m0.conj().T + \
                      full_m1 @ self.quantum_state @ full_m1.conj().T

        # 3. 构造 C 语言中的 reset_0_operator_matrix (带有噪声的注入态)
        # C 语言公式:
        # [ 1-p,  sqrt((1-p)p) ]
        # [ sqrt(p(1-p)), p    ]
        r_matrix = np.array([
            [1.0 - p_reset, np.sqrt((1.0 - p_reset) * p_reset)],
            [np.sqrt(p_reset * (1.0 - p_reset)), p_reset]
        ], dtype=np.complex128)

        # 4. 重新合成全系统密度矩阵 (维度回到 dim x dim)
        # 注意：C 语言是用 Kronecker_Product(trace_state, reset_0)
        # 这里的顺序必须严格匹配 C 语言的 Endianness
        new_rho = np.array([[1.0]], dtype=np.complex128)
        # 按照 C 语言的逻辑，reset 的比特被放回到了 target_qubit 的位置
        # 我们采用分步重组的方式：

        # 简单的实现方式是：构造一个注入算符，将 |0> 态放回 target_qubit
        # 这里直接按照 C 的思路：将 reduced_rho 与 r_matrix 重新张量积
        # 实际上 C 语言通过多次 SWAP 确保了重置比特在 LSB，所以我们也这样做：

        # 更加通用的物理重置公式 (等效于 C 语言的 SWAP+Trace+Insert):
        # rho = rho_reduced \otimes R_at_target
        # 这里我们使用 Kraus 形式来表达这个过程，更不容易出错：
        # K0 = |0_err><0|, K1 = |0_err><1|

        # 构造带有噪声的 |0> 态矢量 (对应 C 语言 reset_0_operator)
        # 注意：C 语言这里的 reset_0_operator 实际上是一个密度矩阵算子

        # 我们直接手动重组：
        self.quantum_state = self._reinsert_qubit(rho_reduced, r_matrix, target_qubit)

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
    np.random.seed()
    def __init__(self, env, parser, ast):
        self.shot_idx = None
        self.shot_total = 0  # 初始化 shot 总数
        self.shots_count_sv = []  # 统计 StateVector 结果
        self.shots_count_dm = []  # 统计 DensityMatrix 结果
        self.env = env
        self.labels = {}
        self.body_instructions = []
        self.source_code_text = parser.get_source_code_text(ast)
        self.labels_info = parser.get_labels_info(ast)
        self.parser = parser

    def evaluate(self, ast):
        try:
            state = self.env.quantum_state
            if state.ndim == 2:  # 密度矩阵
                diag = [np.real(state[i, i]) for i in range(min(4, state.shape[0]))]
            else:  # 状态矢量
                diag = [np.abs(state[i]) ** 2 for i in range(min(4, len(state)))]
        except Exception as e:
            diag = [f"Wait for init: {str(e)}"]
        print(f"  Initial diagonal: {diag}")

        self.env.full_reset()
        self.labels = collect_labels(self, ast)

        # After collect_labels()
        print("Labels → PC mapping:")
        for lab, pc in sorted(self.labels.items()):
            print(f"  {lab:>8} → {pc:3d}")

        print("\nBody instructions (should be sequential 0,1,2,...):")
        for i, instr in enumerate(self.body_instructions):
            op = instr.children[0].value if instr.children else "???"
            print(f"  {i:3d} : {op}")

        # ==============================================
        # 第一阶段：找到 shot 指令（但暂不执行）
        # ==============================================
        shot_instruction = None
        for node in ast.children:
            if node.type == 'Instruction':
                opcode_node = next((c for c in node.children if c.type == 'Opcode'), None)
                if opcode_node and opcode_node.value == 'shot':
                    shot_instruction = node
                    break

        if not shot_instruction:
            raise ValueError("Missing shot instruction")

        num_states = 2 ** self.env.qreg_size
        self.shots_count_sv = [0] * num_states
        self.shots_count_dm = [0] * num_states

        # ==============================================
        # 第二阶段：按源代码顺序执行所有配置指令
        # （qreg / creg / error 执行）
        # ==============================================
        print("=== Executing configuration instructions in source order ===")
        config_opcodes = {'shot', 'qreg', 'creg', 'error'}

        for node in ast.children:
            if node.type != 'Instruction':
                continue
            opcode_node = next((c for c in node.children if c.type == 'Opcode'), None)
            if not opcode_node:
                continue
            opcode = opcode_node.value

            if opcode in config_opcodes:
                print(f"  → Executing config: {opcode}")
                self.execute_instruction(node)

        # ==============================================
        # 第三阶段：从已执行过的 shot 指令中提取 shots 次数
        # ==============================================
        operands_node = next((c for c in shot_instruction.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError("shot instruction missing operands")
        shots = int(operands_node.children[0].value)
        print(f"Number of shots: {shots}")

        # ==============================================
        # 第四阶段：收集主体指令（排除配置指令）
        # ==============================================
        self.body_instructions = []
        for node in ast.children:
            if node.type != 'Instruction':
                continue

            opcode_node = next((c for c in node.children if c.type == 'Opcode'), None)
            if not opcode_node:
                continue
            opcode = opcode_node.value

            # 跳过配置指令和 BEGIN/END
            if opcode in config_opcodes or opcode in ['XQI-BEGIN', 'XQI-END']:
                continue

            self.body_instructions.append(node)

        # 诊断输出：确认收集到的主体指令
        print("\n=== Collected body instructions ===")
        print("Length of body_instructions:", len(self.body_instructions))
        for idx, instr in enumerate(self.body_instructions):
            opcode_node = next((c for c in instr.children if c.type == 'Opcode'), None)
            opcode = opcode_node.value if opcode_node else "NO OPCODE"
            print(f"  {idx:2d}: {opcode}  (node id: {id(instr)})")

        # ==============================================
        # 第五阶段：执行多次 shot
        # ==============================================
        results = []

        for shot_idx in range(shots):
            self.shot_idx = shot_idx + 1
            print(f"\n=== Starting shot {shot_idx + 1}/{shots} ===")

            # 每次 shot 重置量子态、经典寄存器、PC 等
            self.env.reset_for_shot()
            self.env.shot_completed = False
            # if self.env.error_model:  # Apply gate-like noise to all qubits
            #     self.env.apply_quantum_noise(list(range(self.env.qreg_size)))

            # 打印初始对角元（加防护，避免 size=0 崩溃）
            try:
                if self.env.simulation_mode == 'density_matrix' and self.env.qreg_size > 0:
                    diag = [np.real(self.env.quantum_state[i, i]) for i in range(min(4, 2 ** self.env.qreg_size))]
                elif self.env.simulation_mode == 'statevector' and len(self.env.quantum_state) > 0:
                    diag = [np.abs(self.env.quantum_state[i]) ** 2 for i in range(min(4, len(self.env.quantum_state)))]
                else:
                    diag = ["N/A (qreg size=0)"]
            except Exception as e:
                diag = [f"ERROR: {str(e)}"]

            print(f"  Initial diagonal: {diag}")

            # After self.env.reset_for_shot() and noise application
            self.env.pc = 0  # Ensure reset to 0 each shot
            # executed = set()  # Keep this for duplicate prevention
            while self.env.pc < len(self.body_instructions) and not self.env.shot_completed:
                instr_node = self.body_instructions[self.env.pc]
                opcode_node = next((c for c in instr_node.children if c.type == 'Opcode'), None)
                opcode = opcode_node.value if opcode_node else "???"

                print(f"Executing {opcode} at PC={self.env.pc}")

                old_pc = self.env.pc

                self.execute_instruction(instr_node)

                # 如果指令本身没有修改 pc（没有跳转），则自动 +1
                if self.env.pc == old_pc and not self.env.shot_completed:
                    self.env.pc += 1

                # 安全保护
                if self.env.pc < 0 or self.env.pc > len(self.body_instructions) + 100:
                    raise RuntimeError(f"PC out of valid range after {opcode}: {self.env.pc}")
            # 记录 State Vector 的测量结果
            sv_code = self._calculate_current_state_code()
            if sv_code < num_states:
                self.shots_count_sv[sv_code] += 1

            # 记录 Density Matrix 的测量结果
            dm_code = self._calculate_dm_sample_code()
            if dm_code < num_states:
                self.shots_count_dm[dm_code] += 1

            results.append(self.env.creg.copy())
            self.save_final_results(self.shots_count_sv, self.shots_count_dm)
            print(f"  Shot {shot_idx + 1} completed. creg = {self.env.creg}")

        return results  # 如果需要返回所有 shots 的结果

    def _calculate_current_state_code(self):
        """
        将经典寄存器 creg 中的 0/1 序列转换为十进制整数索引
        匹配 C 语言逻辑：Count_Shot_Quantum_State_Code 计算方式
        """
        code = 0
        for i, val in enumerate(self.env.creg):
            # 取实部并四舍五入（处理浮点误差），非零即为 1
            bit = 1 if abs(val.real) > 0.5 else 0
            code += (bit << i)  # 位运算：bit * (2^i)
        return code

    def _calculate_dm_sample_code(self):
        """
        在密度矩阵模式下，测量结果已经由 execute_measure 存入 creg
        这里直接复用计算逻辑（或根据需要定制）
        """
        return self._calculate_current_state_code()

    def save_final_results(self, shots_count_data, shots_count_data_dm):
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()

        # --- 生成 XQI-QC-state.dat ---
        state_path = os.path.join(log_dir, "XQI-QC-state.dat")
        with open(state_path, "wb") as f:
            f.write(b'XQI-QC')  # Tag
            f.write(struct.pack('I', self.env.qreg_size))
            f.write(struct.pack('I', self.shot_total))  # shot次数

            # 此处应有每一步 state 数据的累积，或仅按 C 逻辑写入 Count
            # 写入 COUNT 标识和数据
            f.write(b'COUNT')
            for count in shots_count_data:
                f.write(struct.pack('I', int(count)))

        # --- 生成 XQI-QC-Density-Matrix-state.dat ---
        dm_state_path = os.path.join(log_dir, "XQI-QC-Density-Matrix-state.dat")
        with open(dm_state_path, "wb") as f:
            f.write(b'XQI-QC-Density-Matrix')
            f.write(struct.pack('I', self.env.qreg_size))
            f.write(struct.pack('I', self.shot_total))

            f.write(b'COUNT')
            for count in shots_count_data_dm:
                f.write(struct.pack('I', int(count)))

    def execute_instruction(self, node):
        if node.type == 'Instruction':
            # 提取Opcode和Operands子节点
            opcode_node = None
            operands_node = None
            for child in node.children:
                if child.type == 'Opcode':
                    opcode_node = child
                elif child.type == 'Operands':
                    operands_node = child

            if not opcode_node:
                raise ValueError("Instruction missing opcode")

            opcode = opcode_node.value

            if opcode == 'shot':
                self.execute_shot(node)
            elif opcode == 'error':
                self.execute_error(node)
            elif opcode == 'qreg':
                self.execute_qreg(node)
            elif opcode == 'creg':
                self.execute_creg(node)
            elif opcode == 'MOV':
                self.execute_mov(node)
            elif opcode == 'CNOT':
                self.execute_cnot(node)
            elif opcode == 'U3':
                self.execute_u3(node)
            elif opcode == 'measure':
                self.execute_measure(node)
            elif opcode == 'SUB':
                self.execute_sub(node)
            elif opcode == 'ADD':
                self.execute_add(node)
            elif opcode == 'MUL':
                self.execute_mul(node)
            elif opcode == 'DIV':
                self.execute_div(node)
            elif opcode == 'LDR':
                self.execute_ldr(node)
            elif opcode == 'STR':
                self.execute_str(node)
            elif opcode == 'CLDR':
                self.execute_cldr(node)
            elif opcode == 'CSTR':
                self.execute_cstr(node)
            elif opcode == 'debug':
                self.execute_debug(node)
            elif opcode == 'debug-p':
                self.execute_debug_p(node)
            elif opcode == 'reset':
                self.execute_reset(node)
            elif opcode == 'barrier':
                self.execute_barrier(node)
            elif opcode == 'rand':
                self.execute_rand(node)
            elif opcode == 'GPS':
                self.execute_gps(node)
            elif opcode == 'XQI-BEGIN':
                pass
            elif opcode == 'XQI-END':
                pass
            elif opcode in {'B', 'BL', 'BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'}:
                self.execute_branch(node, opcode)
            else:
                raise ValueError(f"Unknown opcode: {opcode}")

            # 在量子操作后应用噪声
            # if opcode in ['CNOT', 'U3']:
            #     qubits = self._get_affected_qubits(node)
            #     if qubits:
            #         print(f"Unified noise apply for {opcode}")
            #         self.env.apply_quantum_noise(qubits)

            # opcode = opcode_node.value
            # special_mov_dests = {'PC', 'LR', 'SF', 'ZF'}
            # branch_ops = {'B', 'BL', 'BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'}  # 移除 'MOV'
            # if opcode in branch_ops:
            #     pass  # No auto +1
            # elif opcode == 'MOV':
            #     dest_operand = operands_node.children[0].value if operands_node else None
            #     if dest_operand in special_mov_dests:
            #         pass  # No +1 for MOV to PC/LR/etc.
            #     else:
            #         self.env.pc += 1
            # elif opcode in {'ADD', 'SUB', 'MUL', 'DIV', 'U3', 'CNOT', 'reset', 'measure', 'barrier', 'debug',
            #                 'debug-p'}:
            #     self.env.pc += 1

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
            except:
                return []

        elif opcode in ['U3', 'measure', 'reset']:
            # 最后一个操作数是量子位
            if not children:
                return []
            qubit_str = children[-1].value
            try:
                qubit = int(qubit_str[2:-1])
                return [qubit]
            except:
                return []

        return []

    def execute_shot(self, node):
        pass

    def execute_error(self, instruction_node):
        # 从Instruction节点中获取Operands
        operands_node = next((c for c in instruction_node.children if c.type == 'Operands'), None)

        if not operands_node or not operands_node.children:
            raise ValueError("error instruction requires at least the enable parameter")

        operands = operands_node.children
        enable_str = operands[0].value

        if enable_str in ['TRUE', '1']:
            enable = True
        elif enable_str in ['FALSE', '0']:
            enable = False
        else:
            raise ValueError("First parameter must be TRUE/FALSE or 1/0")

        if not enable:
            self.env.error_model = None
            if hasattr(self.env, '_pending_error_model'):
                del self.env._pending_error_model
            return

        code = config.default_Q_error_Code if len(operands) < 2 else int(operands[1].value)
        p1 = config.default_Q1_error_Probability if len(operands) < 3 else float(operands[2].value)
        p2 = config.default_Q2_error_Probability if len(operands) < 4 else float(operands[3].value)
        p_measure = config.default_measure_error_Probability if len(operands) < 5 else float(operands[4].value)
        p_reset = config.default_reset_error_Probability if len(operands) < 6 else float(operands[5].value)

        self.env.error_model = (code, p1, p2, p_measure, p_reset)
        print(f"Error model set: code={code}, p1={p1}, p2={p2}, p_measure={p_measure}, p_reset={p_reset}")

        if hasattr(self.env, '_pending_error_model'):
            self.env._pending_error_model = self.env.error_model

    def execute_qreg(self, node):
        # 获取Operands子节点（关键修正）
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError("qreg instruction missing operands")

        # 从第一个操作数提取字符串
        operand_str = operands_node.children[0].value
        # 解析寄存器大小
        left = operand_str.find('[')
        right = operand_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Invalid qreg format: {operand_str}")
        qreg_size = int(operand_str[left + 1:right])
        self.env.resize_qreg(qreg_size)

    def execute_creg(self, node):
        # 获取Operands子节点
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError("creg instruction missing operands")

        # 从第一个操作数提取字符串
        operand_str = operands_node.children[0].value
        # 解析寄存器大小
        left = operand_str.find('[')
        right = operand_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Invalid creg format: {operand_str}")
        creg_size = int(operand_str[left + 1:right])
        self.env._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.env.creg = self.env._initial_creg.copy()

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
            raise ValueError("MOV instruction requires 2 operands")

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
            except:
                raise ValueError(f"Invalid R register format: {dest_str}")
            if dest_val < 0 or dest_val >= len(self.env.registers):
                raise ValueError(f"R[{dest_val}] out of range")
        else:
            raise ValueError(f"Unsupported destination: {dest_str}")

        # 解析源（支持立即数、R寄存器、PC、LR）
        if src_str in {'PC', 'LR', 'SF', 'ZF'}:
            src_type = src_str
            src_val = None
        elif src_str.startswith('R[') and src_str.endswith(']'):
            src_type = 'R'
            try:
                src_val = int(src_str[2:-1])
            except:
                raise ValueError(f"Invalid source R register: {src_str}")
            if src_val < 0 or src_val >= len(self.env.registers):
                raise ValueError(f"Source R[{src_val}] out of range")
        elif src_str.isdigit() or (src_str.lstrip('-').isdigit()):
            src_type = 'imm'
            src_val = int(src_str)  # 优先尝试整数
        elif '.' in src_str or 'e' in src_str.lower() or 'E' in src_str:
            src_type = 'imm'
            src_val = float(src_str)  # 浮点数
        else:
            raise ValueError(f"Unsupported source operand: {src_str}")

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
                raise ValueError(f"Unsupported src → R dest: {src_type}")

            self.env.registers[dest_val] = value
            # 写入普通 R 寄存器后，更新标志位（模仿算术指令）
            self._set_flags(value)

        elif dest_type == 'PC':
            old_pc = self.env.pc
            if src_type == 'imm' and src_val == 0:
                # 特殊语义：结束当前 shot
                self.env.shot_completed = True
                print("MOV PC,0 detected → marking current shot as completed")
                return
            else:
                # 跳转到 PC
                if src_type == 'R':
                    self.env.pc = int(self.env.registers[src_val])
                elif src_type == 'imm':
                    self.env.pc = int(src_val)
                elif src_type == 'LR':
                    self.env.pc = self.env.lr
                    print(f"MOV PC, LR → jumping to {self.env.pc} (LR value is {self.env.lr})")
                elif src_type in {'SF', 'ZF'}:
                    raise ValueError("Cannot MOV SF/ZF directly to PC")
                else:
                    raise ValueError(f"Unsupported src → PC: {src_type}")
                new_pc = self.env.pc
                print(f"PC ← {src_str}  (value={new_pc}, old_pc was {old_pc}, LR={self.env.lr})")
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
                raise ValueError("Cannot MOV SF/ZF directly to LR")
            else:
                raise ValueError(f"Unsupported src → LR: {src_type}")
            # LR 变更不更新标志位

        elif dest_type == 'SF':
            # 直接设置符号标志
            if src_type == 'imm':
                val = int(src_val)
            elif src_type == 'R':
                val = int(self.env.registers[src_val])
            elif src_type in {'PC', 'LR'}:
                raise ValueError("Cannot MOV PC/LR to SF")
            else:
                val = int(src_val)  # SF/ZF

            self.env.SF = 1 if val != 0 else 0  # 通常 SF=1 表示负数，这里简化处理为非零即1

        elif dest_type == 'ZF':
            # 直接设置零标志
            if src_type == 'imm':
                val = int(src_val)
            elif src_type == 'R':
                val = int(self.env.registers[src_val])
            elif src_type in {'PC', 'LR'}:
                raise ValueError("Cannot MOV PC/LR to ZF")
            else:
                val = int(src_val)

            self.env.ZF = 1 if val == 0 else 0

        else:
            raise ValueError(f"Unsupported destination type: {dest_type}")

        # 调试输出（可选）
        print(f"MOV {dest_str} <- {src_str}  completed.")

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

        # 4. 应用噪声 (如果开启)
        if self.env.error_model:
            p1 = self.env.error_model[1]
            self.env.apply_depolarizing_error(qubit, p1)

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

        # 匹配 C 语言逻辑：CNOT 后对 control 和 target 分别应用 error
        if self.env.error_model:
            p2 = self.env.error_model[2]
            self.env.apply_depolarizing_error(control, p2)
            self.env.apply_depolarizing_error(target, p2)

    def execute_gps(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("GPS requires delta and qubit parameters")

        # 解析delta参数（支持立即数或R寄存器）
        delta = self._parse_parameter(operands_node.children[0], 'R')

        # 验证第二个参数格式（q寄存器）
        self._parse_register_index(operands_node.children[1].value, 'q')

        # 应用全局相位
        phase = np.exp(1j * delta)
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state *= phase
        else:
            self.env.quantum_state = phase * self.env.quantum_state * phase.conj()

        # 应用噪声模型
        if self.env.error_model:
            self.env.apply_quantum_noise([])

    # 辅助方法 ---------------------------------------------------
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

    def execute_measure(self, node):
        assert isinstance(self.env.quantum_state, np.ndarray), \
            f"quantum_state corrupted: {type(self.env.quantum_state)}"

        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("MEASURE requires 2 operands (qubit, creg)")

        qubit = self._parse_register_index(operands_node.children[0].value, 'q')
        creg_idx = self._parse_register_index(operands_node.children[1].value, 'c')

        if qubit >= self.env.qreg_size:
            raise ValueError(f"Qubit index out of range: {qubit}")
        if creg_idx >= len(self.env.creg):
            raise ValueError(f"CReg index out of range: {creg_idx}")

        if self.env.simulation_mode == 'statevector':
            # 1. 计算概率 (基于 LSB: q[0] 是 1, q[1] 是 2...)
            probs = [0.0, 0.0]
            dim = len(self.env.quantum_state)
            for i in range(dim):
                # 获取第 i 个基态在 target qubit 上的比特值 (0 或 1)
                bit = (i >> qubit) & 1
                probs[bit] += np.abs(self.env.quantum_state[i]) ** 2
            # 2. 采样结果
            outcome = np.random.choice([0, 1], p=probs / np.sum(probs))
            # 3. 状态塌缩 (修改此处以匹配位序)
            new_state = np.zeros_like(self.env.quantum_state)
            for i in range(dim):
                if ((i >> qubit) & 1) == outcome:
                    new_state[i] = self.env.quantum_state[i]
            # 4. 归一化
            norm = np.linalg.norm(new_state)
            if norm > 0:
                self.env.quantum_state = new_state / norm
            else:
                self.env.quantum_state = new_state  # 避免除以 0
            # 5. 更新经典寄存器 (匹配 C 语言: 存储 0 或 1，或根据需求存储概率)
            self.env.creg[creg_idx] = outcome

        elif self.env.simulation_mode == 'density_matrix':
            dim = 2 ** self.env.qreg_size
            # 创建对角投影矩阵
            proj0 = np.zeros((dim, dim), dtype=np.complex128)
            proj1 = np.zeros((dim, dim), dtype=np.complex128)

            for i in range(dim):
                # LSB-first: q[0] 是 i 的最低位 (i & 1)
                # qubit 是第几位：(i >> qubit) & 1
                bit = (i >> qubit) & 1
                if bit == 0:
                    proj0[i, i] = 1.0  # 保留 outcome=0
                else:
                    proj1[i, i] = 1.0  # 保留 outcome=1
            proj_ops = [proj0, proj1]

            print("proj0 diagonal:", proj0.diagonal().real)
            print("proj1 diagonal:", proj1.diagonal().real)

            # 计算概率
            probs = np.array([np.real(np.trace(p @ self.env.quantum_state)) for p in proj_ops])
            probs = np.maximum(probs, 0.0)
            total_prob = np.sum(probs)
            print(f"Measure q[{qubit}] raw probs = {probs}, total={total_prob:.12f}")
            if total_prob < 1e-10:
                print("WARNING: total prob near zero! Fallback to outcome=0")
                outcome = 0
                new_rho = proj_ops[0] @ self.env.quantum_state @ proj_ops[0].conj().T
            else:
                probs /= total_prob
                outcome = np.random.choice([0, 1], p=probs)
            proj = proj_ops[outcome]
            new_rho = proj @ self.env.quantum_state @ proj.conj().T
            trace = np.real(np.trace(new_rho))
            print(f"Outcome={outcome}, post-diag={[np.real(new_rho[i, i]) for i in range(dim)]}")
            if trace > 1e-10:
                new_rho /= trace
            else:
                print("Trace too small, fallback to projector")
                new_rho = np.zeros((dim, dim), dtype=np.complex128)
                # 粗暴 fallback 到纯 |outcome> 态（对角 1）
                base_idx = outcome * (dim // 2)  # 简化：假设低位 outcome
                new_rho[base_idx, base_idx] = 1.0
            self.env.quantum_state = new_rho

            if self.env.error_model:
                _, _, _, p_measure, _ = self.env.error_model
                if np.random.rand() < p_measure:
                    outcome = 1 - outcome  # Bit-flip error

            self.env.creg[creg_idx] = outcome

    def execute_add(self, node):
        self._execute_binary_arithmetic(node, 'add')

    def execute_sub(self, node):
        self._execute_binary_arithmetic(node, 'sub')

    def execute_mul(self, node):
        self._execute_binary_arithmetic(node, 'mul')

    def execute_div(self, node):
        self._execute_binary_arithmetic(node, 'div')

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
            raise ValueError(f"Expected {prefix} register, got {reg_str}")
        try:
            index = int(reg_str[len(prefix) + 1:-1])  # 解析类似 q[5] 的格式
        except (ValueError, IndexError):
            raise ValueError(f"Invalid register format: {reg_str}")

        # 寄存器范围检查
        max_size = self.env.qreg_size if prefix == 'q' else \
            len(self.env.registers) if prefix == 'R' else \
                len(self.env.creg)
        if index >= max_size:
            raise ValueError(f"{prefix} register index out of range: {index} (max={max_size - 1})")
        return index

    def _execute_binary_arithmetic(self, node, operation):
        operands = node.children[1].children
        print(f"[{operation.upper()}] raw operands: {[op.value for op in operands]}")

        if len(operands) < 3:
            raise ValueError(f"{operation.upper()} needs 3 operands")

        dest_idx = self._parse_register_index(operands[0].value, 'R')
        src1_parsed = self._parse_operand(operands[1].value)
        src2_parsed = self._parse_operand(operands[2].value)

        print(f"dest: {dest_idx}, src1_parsed: {src1_parsed}, src2_parsed: {src2_parsed}")

        if isinstance(src1_parsed, int) and operands[1].value.startswith('R['):
            val1 = self.env.registers[src1_parsed]
        else:
            val1 = float(src1_parsed)  # 立即数

        if isinstance(src2_parsed, int) and operands[2].value.startswith('R['):
            val2 = self.env.registers[src2_parsed]
        else:
            val2 = float(src2_parsed)

        print(f"values → val1={val1} (from R{src1_parsed}), val2={val2} (from R{src2_parsed})")

        if operation == 'add':
            result = val1 + val2
        elif operation == 'sub':
            result = val1 - val2
        elif operation == 'mul':
            result = val1 * val2
        elif operation == 'div':
            if abs(val2) < 1e-12:
                raise ValueError("Division by zero")
            result = val1 / val2
        else:
            raise ValueError(f"Unknown operation: {operation}")

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
                raise ValueError(f"Cannot parse operand: {operand_str}")

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
            self.env.lr = self.env.pc + 1  # 注意：这里 +1 是返回后下一条指令的地址
            self.env.pc = self._get_label_address(label)
            return

        if opcode == 'BNE':
            if self._condition_met('NE'):
                print(f"  → branch taken to {label} @ {self.labels[label]}")
            else:
                print("  → branch NOT taken (ZF==1)")

        # 条件分支
        if self._condition_met(opcode[1:]):  # BEQ → 'EQ', BNE → 'NE' 等
            self.env.pc = self._get_label_address(label)
        # 不跳转时 → 外层 +1

    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _parse_label_operand(node):
        """从分支指令节点解析标签操作数"""
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError(f"{node.children[0].value} instruction missing label operand")
        label_operand = operands_node.children[0]
        if label_operand.type != 'Label':
            raise ValueError(f"Expected label operand, got {label_operand.type}")
        return label_operand.value.strip(':')  # 去除可能存在的冒号

    def _get_label_address(self, label):
        label = label.strip(':')
        if label not in self.labels:
            raise ValueError(f"Undefined label: {label}")
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

    def execute_ldr(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("LDR requires 2 operands (dest_reg, src_mem)")

        # 解析目标寄存器 R[...]
        dest_reg = self._parse_register_index(operands_node.children[0].value, 'R')

        # 解析源内存地址 M[...]
        src_mem = self._parse_memory_address(operands_node.children[1].value)

        # 执行加载操作
        self.env.registers[dest_reg] = self.env.memory[src_mem]

    def execute_str(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("STR requires 2 operands (src_reg, dest_mem)")

        # 解析源寄存器 R[...]
        src_reg = self._parse_register_index(operands_node.children[0].value, 'R')

        # 解析目标内存地址 M[...]
        dest_mem = self._parse_memory_address(operands_node.children[1].value)

        # 执行存储操作
        self.env.memory[dest_mem] = self.env.registers[src_reg]

    def execute_cldr(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("CLDR requires 3 operands (dest_creg, real_mem, imag_mem)")

        # 解析目标经典寄存器 c[...]
        dest_creg = self._parse_register_index(operands_node.children[0].value, 'c')

        # 解析实部和虚部内存地址
        real_mem = self._parse_memory_address(operands_node.children[1].value)
        imag_mem = self._parse_memory_address(operands_node.children[2].value)

        # 构建复数并存储
        self.env.creg[dest_creg] = complex(
            self.env.memory[real_mem],
            self.env.memory[imag_mem]
        )

    def execute_cstr(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("CSTR requires 3 operands (src_creg, real_mem, imag_mem)")

        # 解析源经典寄存器 c[...]
        src_creg = self._parse_register_index(operands_node.children[0].value, 'c')

        # 解析目标内存地址
        real_mem = self._parse_memory_address(operands_node.children[1].value)
        imag_mem = self._parse_memory_address(operands_node.children[2].value)

        # 分解复数并存储
        complex_val = self.env.creg[src_creg]
        self.env.memory[real_mem] = complex_val.real
        self.env.memory[imag_mem] = complex_val.imag

    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _parse_memory_address(addr_str):
        """解析内存地址格式 M[数字] 或 M[基址+偏移]"""
        if not addr_str.startswith('M'):
            raise ValueError(f"Invalid memory address format: {addr_str}")

        # 提取方括号内容
        left = addr_str.find('[')
        right = addr_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Invalid memory address format: {addr_str}")

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

        # 获取错误模型中的重置概率 (Error Code 8)
        p_reset = 0.0
        if self.env.error_model:
            code, p1, p2, p_measure, p_reset_val = self.env.error_model
            if code == 8:
                p_reset = p1  # C 语言中 reset 使用的是 Q1_error_Probability

        print(f"Executing Physical Reset on q[{qubit}] with p={p_reset}")
        self.env.apply_physical_reset(qubit, p_reset)

    def execute_barrier(self, node):
        # No-op in this simulation
        pass

    def execute_rand(self, node):
        dest = int(node.children[0].value[2:-1])
        seed = int(node.children[1].value[2:-1])
        np.random.seed(int(self.env.registers[seed]))
        self.env.registers[dest] = np.random.uniform(0, 1)

    def print_debug_info(self, shot_id=1):
        if not hasattr(self, '_debug_file_initialized'):
            self._initialize_debug_files()

        q_size = self.env.qreg_size
        dim = 2 ** q_size

        # 模拟数据准备 (State Vector & Density Matrix)
        if self.env.simulation_mode == 'statevector':
            psi = self.env.quantum_state
            rho = np.outer(psi, psi.conj())
        else:
            rho = self.env.quantum_state
            psi = np.sqrt(np.abs(np.diag(rho)))  # 仅用于展示

        # --- 2. 写入 Density Matrix 文本日志 (XQI-QC-Density-Matrix-list.txt) ---
        with open(self.paths['dm_txt'], "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")
            f.write("XQI: current Density Matrix states:\n")
            self._write_matrix_to_text(f, rho)
            self._write_common_debug(f)

        # --- 3. 写入二进制数据 (与 Matlab 接口对接) ---
        # SV Binary: 连续写入实部和虚部 (float32)
        with open(self.paths['sv_dat'], "ab") as f:
            for i in range(dim):
                val = psi[i] if self.env.simulation_mode == 'statevector' else psi[i]
                f.write(struct.pack('ff', float(np.real(val)), float(np.imag(val))))

        # 1. 追加 SV 数据: 连续的 float32 (real, imag)
        with open(self.paths['sv_dat'], "ab") as f:
            for val in psi:
                f.write(struct.pack('ff', float(np.real(val)), float(np.imag(val))))

        # 2. 追加 DM 数据: 按行主序的 float32 (real, imag)
        with open(self.paths['dm_dat'], "ab") as f:
            for i in range(dim):
                for j in range(dim):
                    val = rho[i, j]
                    f.write(struct.pack('ff', float(val.real), float(val.imag)))
    def _initialize_debug_files(self):
        """
        初始化所有调试文件。
        如果是本次运行的第一次 debug，则删除旧文件并写入 Header。
        """
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()

        # 统一定义文件路径
        self.paths = {
            'sv_txt': os.path.join(log_dir, "XQI-QC-list.txt"),
            'sv_dat': os.path.join(log_dir, "XQI-QC-debug.dat"),
            'dm_txt': os.path.join(log_dir, "XQI-QC-Density-Matrix-list.txt"),
            'dm_dat': os.path.join(log_dir, "XQI-QC-Density-Matrix-debug.dat")
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
        header_text += "Label Number   Sequence   Label Symbol\n"
        for idx, (seq, symbol) in enumerate(self.labels_info):
            header_text += f"Label {idx:3d}:     {seq:3d}        {symbol}\n"
        header_text += "\n\n"

        with open(self.paths['sv_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)
        with open(self.paths['dm_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)

        # --- 第三步：初始化二进制文件 (Header) ---
        # 1. SV Binary Header: Tag(6 bytes) + QregSize(4 bytes)
        with open(self.paths['sv_dat'], "wb") as f:
            f.write(b'XQI-QC')  # 对应 C 语言 char[6]
            f.write(struct.pack('I', self.env.qreg_size))  # 对应 C 语言 unsigned int

        # 2. DM Binary Header: Tag(21 bytes) + QregSize(4 bytes)
        with open(self.paths['dm_dat'], "wb") as f:
            tag_dm = b'XQI-QC-Density-Matrix'  # 21字节
            f.write(tag_dm)
            f.write(struct.pack('I', self.env.qreg_size))

        # 标记初始化完成
        self._debug_file_initialized = True



    def _write_matrix_to_text(self, f, matrix):
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
        f.write(f"SIGN_FLAG={self.env.SF};  " if self.env.SF else "SIGN_FLAG=0;  ")
        f.write(f"ZERO_FLAG={self.env.ZF}.\n" if self.env.ZF else "ZERO_FLAG=0.\n")

        f.write("\n memory:\n")
        for idx, val in enumerate(self.env.memory):
            f.write(f"M[{idx:4d}]={val:15.10f}\n")
        f.write("\n\n")