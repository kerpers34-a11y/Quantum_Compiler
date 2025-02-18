import numpy as np
import config
from itertools import product

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
    def __init__(self, qreg_size=0, creg_size=0, max_registers=config.MAX_Register,
                 max_memory=config.MAX_Memory, simulation_mode='statevector'):
        # 参数校验
        if not all(isinstance(x, int) and x >= 0 for x in [qreg_size, creg_size]):
            raise ValueError("Register sizes must be non-negative integers")
        self.qreg_size = qreg_size  # 直接保存为实例变量
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
        self.error_model = None

    def _reset_quantum_register(self, new_size):
        """重置量子寄存器"""
        self.qreg_size = new_size
        self._initial_quantum_state = self._initialize_quantum_state(new_size)
        self.quantum_state = self._initial_quantum_state.copy()

    def resize_qreg(self, new_size):
        """修改量子寄存器大小"""
        if not isinstance(new_size, int) or new_size < 0:
            raise ValueError("qreg_size must be non-negative integer")
        self._reset_quantum_register(new_size)

    def _initialize_quantum_state(self, size):
        """根据模拟模式初始化量子态"""
        if size == 0:
            return np.array([], dtype=np.complex128) if self.simulation_mode == 'statevector' else np.array([], dtype=np.complex128).reshape(0,0)
        if self.simulation_mode == 'statevector':
            state = np.zeros(2 ** size, dtype=np.complex128)
            state[0] = 1.0
        elif self.simulation_mode == 'density_matrix':
            state = np.zeros((2 ** size, 2 ** size), dtype=np.complex128)
            state[0, 0] = 1.0
        else:
            raise ValueError(f"Unsupported simulation mode: {self.simulation_mode}")
        return state

    def reset_for_shot(self):
        """重置量子环境到初始状态"""
        self.quantum_state = self._initial_quantum_state.copy()
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

    def _build_multi_qubit_ops(self, ops, target_qubits):
        """构建作用于指定量子位的多量子位算子"""
        full_ops = []
        dim = 2 ** self.qreg_size

        for op in ops:
            full_op = np.eye(1, dtype=np.complex128)
            for qubit in range(self.qreg_size):
                if qubit in target_qubits:
                    full_op = np.kron(full_op, op)
                else:
                    full_op = np.kron(full_op, np.eye(2))
            full_ops.append(full_op)

        return full_ops

    def reset(self):
        """重置量子环境到初始状态"""
        self.quantum_state = self._initial_quantum_state.copy()
        self.creg = self._initial_creg.copy()
        self.registers.fill(0.0)
        self.memory.fill(0.0)
        self.pc = 0
        self.lr = 0

    def apply_measurement(self, qubit, creg_index):
        """执行量子测量并处理噪声"""
        # 生成投影算子
        proj0 = np.zeros((2 ** self.qreg_size, 2 ** self.qreg_size))
        proj1 = np.zeros_like(proj0)
        for i in range(2 ** self.qreg_size):
            if (i >> qubit) & 1 == 0:
                proj0[i, i] = 1
            else:
                proj1[i, i] = 1

        # 基础测量过程
        if self.simulation_mode == 'statevector':
            prob0 = np.vdot(self.quantum_state, proj0 @ self.quantum_state).real
            if np.random.rand() < prob0:
                result = 0
                new_state = proj0 @ self.quantum_state / np.sqrt(prob0)  # 归一化
            else:
                result = 1
                new_state = proj1 @ self.quantum_state / np.sqrt(1 - prob0)  # 归一化
            self.quantum_state = new_state
        else:
            prob0 = np.trace(proj0 @ self.quantum_state).real
            if np.random.rand() < prob0:
                result = 0
                new_state = (proj0 @ self.quantum_state @ proj0) / prob0
            else:
                result = 1
                new_state = (proj1 @ self.quantum_state @ proj1) / (1 - prob0)
            self.quantum_state = new_state

        # 应用测量噪声模型

        if (self.simulation_mode != 'statevector'
                    and self.error_model
                    and self.error_model[0] == 9):
            p_error = self.error_model[1]  # 单比特错误概率
            if np.random.rand() < p_error:
                result ^= 1  # 翻转测量结果

        # 存储结果（保持复数形式兼容性）
        self.creg[creg_index] = complex(result)

    def apply_quantum_noise(self, qubits):
        """应用当前配置的量子噪声"""
        if not self.error_model or self.error_model[0] == 9:
            return  # 测量错误单独处理

        noise_type, *params = self.error_model
        kraus_ops = self.generate_kraus_operators(noise_type, qubits, params)

        if self.simulation_mode == 'statevector':
            # 计算各Kraus算子的概率
            probs = np.array([np.linalg.norm(k_op @ self.quantum_state) ** 2 for k_op in kraus_ops])
            probs /= np.sum(probs)  # 归一化
            # 随机选择并应用
            chosen = np.random.choice(len(kraus_ops), p=probs)
            new_state = kraus_ops[chosen] @ self.quantum_state
            norm = np.linalg.norm(new_state)
            self.quantum_state = new_state / norm if norm > 0 else new_state
        else:
            self.quantum_state = sum(k @ self.quantum_state @ k.conj().T for k in kraus_ops)

    def generate_kraus_operators(self, noise_type, qubits, params):
        """生成对应噪声模型的Kraus算子"""
        operators = []
        # 解极化噪声 (类型1)
        if noise_type == 1:
            p = params[0] if params else config.default_Q1_error_Probability
            basis = [np.eye(2),
                     np.array([[0, 1], [1, 0]]),
                     np.array([[0, -1j], [1j, 0]]),
                     np.array([[1, 0], [0, -1]])]
            per_qubit_ops = [
                [np.sqrt(1 - p) * basis[0]] +
                [np.sqrt(p / 3) * b for b in basis[1:]]
                for _ in qubits
            ]
            for combo in product(*per_qubit_ops):
                # 构造每个量子位的操作列表
                op_list = [np.eye(2) for _ in range(self.qreg_size)]
                for q, op in zip(qubits, combo):
                    op_list[q] = op
                # 按最高位到最低位构造张量积
                current_op = None
                for q_idx in reversed(range(self.qreg_size)):
                    if current_op is None:
                        current_op = op_list[q_idx]
                    else:
                        current_op = np.kron(op_list[q_idx], current_op)
                operators.append(current_op)

        # 幅度阻尼 (类型2)
        elif noise_type == 2:
            gamma = params[0] if params else config.default_amp_damping_gamma
            per_qubit_ops = []
            for q in qubits:
                k0 = np.array([[1, 0],
                               [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
                k1 = np.array([[0, np.sqrt(gamma)],
                               [0, 0]], dtype=np.complex128)
                per_qubit_ops.append([k0, k1])

            # 生成所有可能的算子组合
            for combo in product(*per_qubit_ops):
                # 初始化操作列表（按qreg_size顺序）
                op_list = [np.eye(2) for _ in range(self.qreg_size)]
                # 填充当前组合的算子
                for q, op in zip(qubits, combo):
                    op_list[q] = op

                # 构建全系统算子（从最高位到最低位）
                current_op = None
                for q_idx in reversed(range(self.qreg_size)):
                    if current_op is None:
                        current_op = op_list[q_idx]
                    else:
                        current_op = np.kron(op_list[q_idx], current_op)

                operators.append(current_op)

        # 相位阻尼 (类型3)
        elif noise_type == 3:
            gamma = params[0] if params else config.default_phase_damping_gamma
            per_qubit_ops = []
            for q in qubits:
                k0 = np.array([[1, 0],
                               [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
                k1 = np.array([[0, 0],
                               [0, np.sqrt(gamma)]], dtype=np.complex128)
                per_qubit_ops.append([k0, k1])

            for combo in product(*per_qubit_ops):
                op_list = [np.eye(2) for _ in range(self.qreg_size)]
                for q, op in zip(qubits, combo):
                    op_list[q] = op

                # 张量积构建（注意量子位顺序）
                current_op = None
                for q_idx in reversed(range(self.qreg_size)):
                    current_op = op_list[q_idx] if current_op is None \
                        else np.kron(op_list[q_idx], current_op)

                operators.append(current_op)

        # 重置误差 (类型8)
        elif noise_type == 8:
            p = params[0] if params else config.default_reset_error_Probability
            per_qubit_ops = []
            for q in qubits:
                # 重置到|0>和|1>的概率
                k0 = np.sqrt(1 - p) * np.eye(2)
                k1 = np.sqrt(p) * np.array([[1, 1], [0, 0]])  # 重置到|0>
                k2 = np.sqrt(p) * np.array([[0, 0], [1, 1]])  # 重置到|1>
                per_qubit_ops.append([k0, k1, k2])

            for combo in product(*per_qubit_ops):
                op_list = [np.eye(2) for _ in range(self.qreg_size)]
                for q, op in zip(qubits, combo):
                    op_list[q] = op

                # 逆序构建张量积
                full_op = None
                for q_idx in reversed(range(self.qreg_size)):
                    if full_op is None:
                        full_op = op_list[q_idx]
                    else:
                        full_op = np.kron(op_list[q_idx], full_op)

                operators.append(full_op)

        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        self.validate_kraus_ops(operators)
        return operators

    @staticmethod
    def validate_kraus_ops(operators):
        """验证Σ Ki†Ki = I 的完备性关系"""
        # 处理空操作符列表的特殊情况
        if not operators:
            raise ValueError("Kraus operators list cannot be empty")

        # 获取维度信息并校验矩阵形状
        first_op = operators[0]
        if not isinstance(first_op, np.ndarray):
            raise TypeError("Kraus operators must be numpy arrays")
        dim = first_op.shape[0]
        if first_op.ndim != 2 or dim != first_op.shape[1]:
            raise ValueError("Kraus operators must be square matrices")

        # 计算总和并校验维度一致性
        sum_product = np.zeros((dim, dim), dtype=np.complex128)  # 确保 sum_product 始终被初始化
        if dim < 1024:
            for op in operators:
                if op.shape != (dim, dim):
                    raise ValueError(f"Operator dimension mismatch: expected ({dim},{dim}), got {op.shape}")
                sum_product += op.conj().T @ op
        else:
            block_size = 256
            for i in range(0, dim, block_size):
                for j in range(0, dim, block_size):
                    block = sum(
                        op.conj().T[i:i + block_size, j:j + block_size] @ op[i:i + block_size, j:j + block_size]
                        for op in operators)
                    sum_product[i:i + block_size, j:j + block_size] = block

        # 生成单位矩阵并设置合理容差
        identity = np.eye(dim, dtype=np.complex128)
        atol = max(1e-12, np.finfo(sum_product.dtype).eps * 1e4)  # 自适应浮点精度

        if not np.allclose(sum_product, identity, rtol=float(0), atol=float(atol)):
            max_error = np.max(np.abs(sum_product - identity))
            raise ValueError(
                f"Kraus operators violate completeness relation\n"
                f"Max error: {max_error:.2e}\n"
                f"Allowed tolerance: {atol:.2e}"
            )

class Evaluator:
    def __init__(self, env):
        self.env = env
        self.labels = {}
        self.body_instructions = []

    def evaluate(self, ast):
        self.env.full_reset()
        self.labels = collect_labels(ast)

        # 分割AST为配置部分和主体部分
        shot_node = None
        config_nodes = []
        found_shot = False

        # 第一次遍历：定位shot节点和配置指令
        for node in ast.children:
            if isinstance(node, ASTNode) and node.type == 'Opcode' and node.value[0] == 'shot':
                shot_node = node
                found_shot = True
                continue

            if found_shot:
                if node.type == 'Opcode' and node.value[0] in ['error', 'qreg', 'creg']:
                    config_nodes.append(node)
                else:
                    self.body_instructions.append(node)

        if not shot_node:
            raise ValueError("Missing shot instruction")

        # 处理配置前的节点（如XQI-BEGIN）
        for node in ast.children:
            if node is shot_node: break
            self.execute_instruction(node)

        # 处理配置指令
        for node in config_nodes:
            self.execute_instruction(node)

        # 执行shot循环
        shots = int(shot_node.children[0].value[0])
        results = []
        for _ in range(shots):
            self.env.reset_for_shot()
            for node in self.body_instructions:
                if isinstance(node, ASTNode):
                    self.execute_instruction(node)
            results.append(self.env.creg.copy())

        print(f"Results after {shots} shots: {results}")

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
            elif opcode == 'BEQ':
                self.execute_beq(node)
            elif opcode == 'BGT':
                self.execute_bgt(node)
            elif opcode == 'BGE':
                self.execute_bge(node)
            elif opcode == 'BLT':
                self.execute_blt(node)
            elif opcode == 'BLE':
                self.execute_ble(node)
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
            else:
                raise ValueError(f"Unknown opcode: {opcode}")
        # 在量子操作后应用噪声
        if node.type == 'Opcode' and node.value[0] in ['CNOT', 'U3']:
            qubits = self._get_affected_qubits(node)
            self.env.apply_quantum_noise(qubits)

        # 分支指令列表
        branch_ops = {'B', 'BL', 'BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'}
        if node.type == 'Opcode' and node.value[0] in branch_ops:
            pass  # 分支指令已显式设置PC
        else:
            self.env.pc += 1  # 非分支指令自动递增

    @staticmethod
    def _get_affected_qubits(node):
        """获取指令影响的量子位"""
        opcode = node.value[0]
        if opcode == 'CNOT':
            return [int(node.children[0].value[0][1:-1]),
                    int(node.children[1].value[0][1:-1])]
        elif opcode == 'U3' or opcode == 'measure':
            return [int(node.children[-1].value[0][1:-1])]
        return []

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
        params = [float(c.value[0]) for c in node.children[1:]]
        self.env.error_model = (error_type, *params)

    def execute_qreg(self, node):
        qreg_size = int(node.children[0].value[0][1:-1])
        self.env.qreg_size = qreg_size  # 更新实例变量
        #self.env._initial_quantum_state = self.env._initialize_quantum_state(qreg_size)
        #self.env.quantum_state = self.env._initial_quantum_state.copy()
        self.env.resize_qreg(qreg_size)

    def execute_creg(self, node):
        creg_size = int(node.children[0].value[0][1:-1])
        self.env._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.env.creg = self.env._initial_creg.copy()

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

        # 构建CNOT门（控制位为高位的情况）
        full_op = np.zeros((2 ** self.env.qreg_size, 2 ** self.env.qreg_size))
        for i in range(2 ** self.env.qreg_size):
            if (i >> control) & 1:  # 控制位为1
                j = i ^ (1 << target)  # 翻转目标位
                full_op[j, i] = 1
            else:
                full_op[i, i] = 1

        # 应用门操作
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = full_op @ self.env.quantum_state
        else:
            self.env.quantum_state = full_op @ self.env.quantum_state @ full_op.conj().T

        # 应用噪声
        if self.env.error_model:
            self.env.apply_quantum_noise([control, target])

    def execute_u3(self, node):
        theta = float(node.children[0].value[0])
        phi = float(node.children[1].value[0])
        lambda_ = float(node.children[2].value[0])
        qubit = int(node.children[3].value[0][1:-1])

        # 使用张量积构建方式
        if qubit >= self.env.qreg_size:
            raise ValueError(f"Qubit index out of range: {qubit}")

        # 单量子比特U3门
        u3 = np.array([
            [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lambda_)) * np.cos(theta / 2)]
        ])

        # 构建全系统矩阵
        full_op = 1
        for q in reversed(range(self.env.qreg_size)):
            if q == qubit:
                full_op = np.kron(full_op, u3)
            else:
                full_op = np.kron(full_op, np.eye(2))

        # 应用操作
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = full_op @ self.env.quantum_state
        else:
            self.env.quantum_state = full_op @ self.env.quantum_state @ full_op.conj().T

    def execute_measure(self, node):
        qubit = int(node.children[0].value[0][1:-1])
        creg_idx = int(node.children[2].value[0][1:-1])

        # 生成投影算子
        proj0 = np.zeros((2 ** self.env.qreg_size, 2 ** self.env.qreg_size))
        proj1 = np.zeros_like(proj0)
        for i in range(2 ** self.env.qreg_size):
            if (i >> qubit) & 1 == 0:
                proj0[i, i] = 1
            else:
                proj1[i, i] = 1

        if self.env.simulation_mode == 'statevector':
            # 态矢量测量
            prob0 = np.real(np.vdot(self.env.quantum_state, proj0 @ self.env.quantum_state))
            if np.random.rand() < prob0:
                self.env.quantum_state = (proj0 @ self.env.quantum_state) / np.sqrt(prob0)
                result = 0
            else:
                self.env.quantum_state = (proj1 @ self.env.quantum_state) / np.sqrt(1 - prob0)
                result = 1
        else:
            # 密度矩阵测量
            prob0 = np.trace(proj0 @ self.env.quantum_state).real
            if np.random.rand() < prob0:
                self.env.quantum_state = (proj0 @ self.env.quantum_state @ proj0) / prob0
                result = 0
            else:
                self.env.quantum_state = (proj1 @ self.env.quantum_state @ proj1) / (1 - prob0)
                result = 1

        self.env.creg[creg_idx] = result  # 存储经典结果

    def execute_add(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        result = self.env.registers[src1] + self.env.registers[src2]
        self.env.registers[dest] = result
        self.env.SF = 1 if result < 0 else 0
        self.env.ZF = 1 if result == 0 else 0

    def execute_sub(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        result = self.env.registers[src1] - self.env.registers[src2]
        self.env.registers[dest] = result
        self.env.SF = 1 if result < 0 else 0
        self.env.ZF = 1 if result == 0 else 0

    def execute_mul(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        result = self.env.registers[src1] * self.env.registers[src2]
        self.env.registers[dest] = result
        self.env.SF = 1 if result < 0 else 0
        self.env.ZF = 1 if result == 0 else 0

    def execute_div(self, node):
        dest = int(node.children[0].value[0][2:-1])
        src1 = int(node.children[1].value[0][2:-1])
        src2 = int(node.children[2].value[0][2:-1])
        result = self.env.registers[src1] // self.env.registers[src2]  # 根据需求处理除法
        self.env.registers[dest] = result
        self.env.SF = 1 if result < 0 else 0
        self.env.ZF = 1 if result == 0 else 0

    def execute_b(self, node):
        label = node.children[0].value[0]
        self.env.pc = self.labels[label]

    def execute_bl(self, node):
        label = node.children[0].value[0]
        self.env.lr = self.env.pc + 1  # 保存下一条指令地址
        self.env.pc = self.labels[label]  # 跳转

    def execute_beq(self, node):
        label = node.children[0].value[0]
        if self.env.ZF == 1:
            self.env.pc = self.labels[label]

    def execute_bne(self, node):
        label = node.children[0].value[0]
        if self.env.ZF == 0:
            self.env.pc = self.labels[label]

    def execute_bgt(self, node):
        label = node.children[0].value[0]
        if self.env.SF == 0 and self.env.ZF == 0:
            self.env.pc = self.labels[label]

    def execute_bge(self, node):
        label = node.children[0].value[0]
        if self.env.SF == 0 or self.env.ZF == 1:
            self.env.pc = self.labels[label]

    def execute_blt(self, node):
        label = node.children[0].value[0]
        if self.env.SF == 1 and self.env.ZF == 0:
            self.env.pc = self.labels[label]

    def execute_ble(self, node):
        label = node.children[0].value[0]
        if self.env.SF == 1 or self.env.ZF == 1:
            self.env.pc = self.labels[label]

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
        self.env.quantum_state = np.zeros(2 ** self.env.qreg_size, dtype=np.complex128)
        self.env.quantum_state[0] = 1.0

        # 应用噪声
        if self.env.error_model:
            self.env.apply_quantum_noise([qubit])

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
        phase = np.exp(1j * delta)
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state *= phase
        else:
            self.env.quantum_state = phase * self.env.quantum_state * np.conj(phase)
        # 应用噪声
        if self.env.error_model:
            self.env.apply_quantum_noise([])  # 全局噪声

    def print_debug_info(self):
        print("Quantum Register State:")
        print(self.env.quantum_state)

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

