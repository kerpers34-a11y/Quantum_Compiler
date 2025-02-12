import numpy as np
import config

class ASTNode:
    def __init__(self, type, value=None, children=None, line=None, col=None):
        self.type = type
        self.value = value if value is not None else []
        self.children = children if children is not None else []
        self.line = line
        self.col = col

def collect_labels(ast):
    labels = {}
    pc = 0
    for node in ast.children:
        if node.type == 'Label':
            label = node.value[0]
            labels[label] = pc
        elif node.type == 'Instruction':
            pc += 1
    return labels


class QuantumEnvironment:
    def __init__(self, qreg_size, creg_size, max_registers, max_memory):
        self.qreg_size = qreg_size
        self.creg_size = creg_size
        self.max_registers = max_registers
        self.max_memory = max_memory

        # 量子寄存器 qreg，初始状态为 |00...0>
        self.qreg = np.zeros(2 ** qreg_size, dtype=np.complex128)
        self.qreg[0] = 1.0

        # 经典寄存器 c[n]，初始状态为 0
        self.creg = [np.complex128(0) for _ in range(creg_size)]

        # 通用寄存器 R[n]，初始状态为 0
        self.registers = [0.0 for _ in range(max_registers)]

        # 经典存储单元 M[n]，初始状态为 0
        self.memory = [0.0 for _ in range(max_memory)]

        # 程序计数器 PC
        self.pc = 0

        # 链接寄存器 LR
        self.lr = 0

        # 密度矩阵，初始状态为 |00...0><00...0|
        self.density_matrix = np.outer(self.qreg, np.conj(self.qreg))

        # 误差模型
        self.error_model = None

        # 保存初始状态
        self.initial_qreg = self.qreg.copy()
        self.initial_creg = self.creg.copy()

    def reset(self):
        self.qreg = self.initial_qreg.copy()
        self.creg = self.initial_creg.copy()
        self.registers = [0.0 for _ in range(self.max_registers)]
        self.memory = [0.0 for _ in range(self.max_memory)]
        self.pc = 0
        self.lr = 0
        self.density_matrix = np.outer(self.qreg, np.conj(self.qreg))
        self.error_model = None

    def apply_noise(self, error_type, qubits):
        if error_type == 0:
            return  # 不应用任何噪声
        if error_type == 1:  # 解极化误差
            self.apply_depolarizing_error(qubits)
        elif error_type == 2:  # 幅度衰减误差
            self.apply_amplitude_damping_error(qubits)
        elif error_type == 3:  # 相位衰减误差
            self.apply_phase_damping_error(qubits)
        elif error_type == 8:  # 重置误差
            self.apply_reset_error(qubits)
        elif error_type == 9:  # 测量误差
            self.apply_measure_error(qubits)

    def apply_depolarizing_error(self, qubits):
        p = config.default_Q1_error_Probability if len(qubits) == 1 else config.default_Q2_error_Probability
        for qubit in qubits:
            if np.random.rand() < p:
                self.qreg = self.depolarize(self.qreg, qubit)

    def apply_amplitude_damping_error(self, qubits):
        gamma = 0.01  # 幅度衰减参数
        for qubit in qubits:
            self.qreg = self.amplitude_damping(self.qreg, qubit, gamma)

    def apply_phase_damping_error(self, qubits):
        gamma = 0.01  # 相位衰减参数
        for qubit in qubits:
            self.qreg = self.phase_damping(self.qreg, qubit, gamma)

    def apply_reset_error(self, qubits):
        p = config.default_reset_error_Probability
        for qubit in qubits:
            if np.random.rand() < p:
                self.qreg = self.reset_qubit(self.qreg, qubit)

    def apply_measure_error(self, qubits):
        p = config.default_measure_error_Probability
        for qubit in qubits:
            if np.random.rand() < p:
                self.qreg = self.flip_measurement(self.qreg, qubit)

    def depolarize(self, state, qubit):
        dim = 2 ** self.qreg_size
        identity = np.eye(dim, dtype=np.complex128)
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        p = 1 / (1 + 3 * (1 - 1 / (2 ** self.qreg_size)))
        state = p * state + (1 - p) * (identity + pauli_x + pauli_y + pauli_z) @ state
        return state

    def amplitude_damping(self, state, qubit, gamma):
        dim = 2 ** self.qreg_size
        kraus_ops = [
            np.eye(2, dtype=np.complex128),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        ]

        for op in kraus_ops:
            op_full = np.eye(dim, dtype=np.complex128)
            op_full[qubit * 2:(qubit + 1) * 2, qubit * 2:(qubit + 1) * 2] = op
            state = op_full @ state
        return state

    def phase_damping(self, state, qubit, gamma):
        dim = 2 ** self.qreg_size
        kraus_ops = [
            np.eye(2, dtype=np.complex128),
            np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)
        ]

        for op in kraus_ops:
            op_full = np.eye(dim, dtype=np.complex128)
            op_full[qubit * 2:(qubit + 1) * 2, qubit * 2:(qubit + 1) * 2] = op
            state = op_full @ state
        return state

    def reset_qubit(self, state, qubit):
        dim = 2 ** self.qreg_size
        reset_op = np.zeros((dim, dim), dtype=np.complex128)
        reset_op[0, 0] = 1
        reset_op[qubit * 2, qubit * 2] = 1
        return reset_op @ state

    def flip_measurement(self, state, qubit):
        dim = 2 ** self.qreg_size
        flip_op = np.eye(dim, dtype=np.complex128)
        flip_op[qubit * 2, qubit * 2] = 0
        flip_op[qubit * 2 + 1, qubit * 2 + 1] = 0
        flip_op[qubit * 2, qubit * 2 + 1] = 1
        flip_op[qubit * 2 + 1, qubit * 2] = 1
        return flip_op @ state


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.labels = {}

    def evaluate(self, ast):
        self.env.reset()
        self.labels = collect_labels(ast)
        for node in ast.children:
            if isinstance(node, ASTNode):
                self.execute_instruction(node)

    def execute_instruction(self, node):
        if node.type == 'Opcode':
            opcode = node.value[0]
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
            elif opcode == 'B':
                self.execute_b(node)
            elif opcode == 'BL':
                self.execute_bl(node)
            elif opcode == 'BNE':
                self.execute_bne(node)
            elif opcode == 'SUB':
                self.execute_sub(node)
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
            else:
                raise ValueError(f"Unknown opcode: {opcode}")

    def execute_shot(self, node):
        shots = int(node.children[0].value[0])
        results = []
        for _ in range(shots):
            self.env.reset()
            for node in node.parent.children:
                if isinstance(node, ASTNode) and node.type == 'Opcode':
                    self.execute_instruction(node)
            results.append(self.env.creg)
        print(f"Results after {shots} shots: {results}")

    def execute_error(self, node):
        error_type = int(node.children[0].value[0])
        self.env.error_model = error_type

    def execute_qreg(self, node):
        qreg_size = int(node.children[0].value[0][1:-1])
        self.env.qreg = np.zeros(2 ** qreg_size, dtype=np.complex128)
        self.env.qreg[0] = 1.0

    def execute_creg(self, node):
        creg_size = int(node.children[0].value[0][1:-1])
        self.env.creg = [np.complex128(0) for _ in range(creg_size)]

    def execute_mov(self, node):
        dest = node.children[0].value[0]
        src = node.children[1].value[0]
        if dest.startswith('R[') and src.startswith('R['):
            dest_index = int(dest[2:-1])
            src_index = int(src[2:-1])
            self.env.registers[dest_index] = self.env.registers[src_index]
        elif dest.startswith('R[') and src.startswith('PC'):
            dest_index = int(dest[2:-1])
            self.env.registers[dest_index] = self.env.pc
        elif dest.startswith('R[') and src.startswith('LR'):
            dest_index = int(dest[2:-1])
            self.env.registers[dest_index] = self.env.lr
        elif dest.startswith('PC') and src.startswith('R['):
            src_index = int(src[2:-1])
            self.env.pc = self.env.registers[src_index]
        elif dest.startswith('LR') and src.startswith('R['):
            src_index = int(src[2:-1])
            self.env.lr = self.env.registers[src_index]
        elif dest.startswith('R[') and src.isdigit():
            dest_index = int(dest[2:-1])
            self.env.registers[dest_index] = float(src)
        else:
            raise ValueError(f"Invalid MOV instruction: {node}")

    def execute_cnot(self, node):
        control = int(node.children[0].value[0][1:-1])
        target = int(node.children[1].value[0][1:-1])

        # 构建 CNOT 门的矩阵
        cnot_matrix = np.eye(2 ** self.env.qreg_size, dtype=np.complex128)
        for i in range(2 ** (self.env.qreg_size - 1)):
            cnot_matrix[2 * i + 1, 2 * i + 1] = 0
            cnot_matrix[2 * i + 1, 2 * i + 2] = 1
            cnot_matrix[2 * i + 2, 2 * i + 1] = 1
            cnot_matrix[2 * i + 2, 2 * i + 2] = 0

        # 应用 CNOT 门
        self.env.qreg = cnot_matrix @ self.env.qreg

        # 应用噪声
        if self.env.error_model:
            self.env.apply_noise(self.env.error_model, [control, target])

    def execute_u3(self, node):
        theta = float(node.children[0].value[0])
        phi = float(node.children[1].value[0])
        lambda_ = float(node.children[2].value[0])
        qubit = int(node.children[3].value[0][1:-1])

        # 构建 U3 门的矩阵
        u3_matrix = np.array([
            [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lambda_)) * np.cos(theta / 2)]
        ], dtype=np.complex128)

        # 构建全系统的 U3 门
        full_u3_matrix = np.eye(2 ** self.env.qreg_size, dtype=np.complex128)
        for i in range(2 ** (self.env.qreg_size - 1)):
            if (i >> qubit) & 1 == 0:
                full_u3_matrix[2 * i, 2 * i] = u3_matrix[0, 0]
                full_u3_matrix[2 * i, 2 * i + 1] = u3_matrix[0, 1]
                full_u3_matrix[2 * i + 1, 2 * i] = u3_matrix[1, 0]
                full_u3_matrix[2 * i + 1, 2 * i + 1] = u3_matrix[1, 1]

        # 应用 U3 门
        self.env.qreg = full_u3_matrix @ self.env.qreg

        # 应用噪声
        if self.env.error_model:
            self.env.apply_noise(self.env.error_model, [qubit])

    def execute_measure(self, node):
        qubit = int(node.children[0].value[0][1:-1])
        creg = int(node.children[2].value[0][1:-1])

        # 计算测量的概率
        probabilities = np.abs(self.env.qreg) ** 2
        result = np.random.choice([0, 1], p=[np.sum(probabilities[::2]), np.sum(probabilities[1::2])])

        # 更新量子态
        if result == 0:
            self.env.qreg = self.env.qreg[::2] / np.sqrt(np.sum(probabilities[::2]))
        else:
            self.env.qreg = self.env.qreg[1::2] / np.sqrt(np.sum(probabilities[1::2]))

        # 更新经典寄存器
        self.env.creg[creg] = result

        # 应用噪声
        if self.env.error_model:
            self.env.apply_noise(self.env.error_model, [qubit])

    def execute_b(self, node):
        label = node.children[0].value[0]
        self.env.pc = self.labels[label]

    def execute_bl(self, node):
        label = node.children[0].value[0]
        self.env.lr = self.env.pc + 1
        self.env.pc = self.labels[label]

    def execute_bne(self, node):
        label = node.children[0].value[0]
        if self.env.registers[15] != self.env.registers[14]:
            self.env.pc = self.labels[label]

    def execute_sub(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        self.env.registers[dest] = self.env.registers[src1] - self.env.registers[src2]

    def execute_ldr(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src = int(node.children[1].value[0][2:-1])
        self.env.registers[dest] = self.env.memory[src]

    def execute_str(self, node):
        src = int(node.children[0].value[0][2:-1])
        dest = int(node.children[1].value[0][2:-1])
        self.env.memory[dest] = self.env.registers[src]

    def execute_cldr(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        self.env.creg[dest] = complex(self.env.memory[src1], self.env.memory[src2])

    def execute_cstr(self, node):
        src = int(node.children[0].value[0][2:-1])
        dest1 = int(node.children[1].value[0][2:-1])
        dest2 = int(node.children[2].value[0][2:-1])
        self.env.memory[dest1] = self.env.creg[src].real
        self.env.memory[dest2] = self.env.creg[src].imag

    def execute_debug(self, node):
        self.print_debug_info()

    def execute_debug_p(self, node):
        self.print_debug_info()
        input("Press 'p' to continue: ")

    def execute_reset(self, node):
        qubit = int(node.children[0].value[0][1:-1])
        self.env.qreg = np.zeros(2 ** self.env.qreg_size, dtype=np.complex128)
        self.env.qreg[0] = 1.0

        # 应用噪声
        if self.env.error_model:
            self.env.apply_noise(self.env.error_model, [qubit])

    def execute_barrier(self, node):
        # No-op in this simulation
        pass

    def execute_rand(self, node):
        dest = int(node.children[0].value[0][2:-1])
        seed = int(node.children[1].value[0][2:-1])
        np.random.seed(int(self.env.registers[seed]))
        self.env.registers[dest] = np.random.uniform(0, 1)

    def execute_gps(self, node):
        delta = float(node.children[0].value[0])
        qubit = int(node.children[1].value[0][1:-1])

        # 构建全局相位门的矩阵
        gps_matrix = np.eye(2 ** self.env.qreg_size, dtype=np.complex128)
        for i in range(2 ** (self.env.qreg_size - 1)):
            if (i >> qubit) & 1 == 1:
                gps_matrix[i, i] = np.exp(1j * delta)

        # 应用全局相位门
        self.env.qreg = gps_matrix @ self.env.qreg

        # 应用噪声
        if self.env.error_model:
            self.env.apply_noise(self.env.error_model, [qubit])

    def print_debug_info(self):
        print("Quantum Register State:")
        print(self.env.qreg)

        print("Classical Register State:")
        for i, creg in enumerate(self.env.creg):
            print(f"c[{i}]: {creg}")

        print("General Purpose Registers:")
        for i, reg in enumerate(self.env.registers):
            print(f"R[{i}]: {reg}")

        print("Memory:")
        for i, mem in enumerate(self.env.memory):
            print(f"M[{i}]: {mem}")

        print("Program Counter (PC):", self.env.pc)
        print("Link Register (LR):", self.env.lr)
        print("Density Matrix:")
        print(self.env.density_matrix)

