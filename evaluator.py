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

class InstructionError(ValueError):
    def __init__(self, instr_name, msg):
        super().__init__(f"[{instr_name}] {msg}")

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
        """重置量子及CPSR环境到初始状态"""
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
        shot_instruction = None
        config_instructions = []  # 存储所有配置指令（error/qreg/creg/shot）

        # 第一阶段：遍历所有节点收集配置指令
        for node in ast.children:
            if node.type == 'Instruction':
                for child in node.children:
                    if child.type == 'Opcode':
                        opcode = child.value[0]
                        # 收集所有配置指令（包括shot）
                        if opcode in ['error', 'qreg', 'creg', 'shot']:
                            if node not in config_instructions:
                                config_instructions.append(node)
                        if opcode == 'shot':
                            shot_instruction = node

        # 验证必须存在shot指令
        if not shot_instruction:
            raise ValueError("Missing shot instruction")

        # 第二阶段：执行所有配置指令（包括error/qreg/creg/shot）
        for instr_node in config_instructions:
            self.execute_instruction(instr_node)  # 这里会触发execute_error的打印

        # 第三阶段：提取shots参数
        operands_node = next((c for c in shot_instruction.children if c.type == 'Operands'), None)
        shots = int(operands_node.children[0].value[0])  # 从已执行的shot指令中获取参数

        # 第四阶段：收集body指令（排除所有配置指令）
        self.body_instructions = [
            node for node in ast.children
            if node.type == 'Instruction' and node not in config_instructions
        ]

        # 第五阶段：执行shot循环
        results = []
        for _ in range(shots):
            self.env.reset_for_shot()
            # 执行所有body指令
            for instr_node in self.body_instructions:
                self.execute_instruction(instr_node)
            results.append(self.env.creg.copy())

        print(f"\nResults after {shots} shots: {results}")

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

            opcode = opcode_node.value[0]

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
            elif opcode == 'XQI-BEGIN':
                pass
            elif opcode == 'XQI-END':
                pass
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
        pass

    def execute_error(self, instruction_node):
        # 从Instruction节点中获取Operands
        print(f"\n123{instruction_node}")
        operands_node = next((c for c in instruction_node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("Invalid error instruction format")

        # 提取第一个操作数作为error_type
        error_type_operand = operands_node.children[0]
        error_type = int(error_type_operand.value[0])
        print(f"\nerror_type={error_type}")

        # 提取后续操作数作为参数
        params = [float(op.value[0]) for op in operands_node.children[1:]]

        # 设置错误模型
        self.env.error_model = (error_type, *params)

    def execute_qreg(self, node):
        # 获取Operands子节点（关键修正）
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError("qreg instruction missing operands")

        # 从第一个操作数提取字符串
        operand_str = operands_node.children[0].value[0]

        # 解析寄存器大小
        left = operand_str.find('[')
        right = operand_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Invalid qreg format: {operand_str}")

        qreg_size = int(operand_str[left + 1:right])
        self.env.resize_qreg(qreg_size)

    def execute_creg(self, node):
        # 获取Operands子节点（关键修正）
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError("creg instruction missing operands")

        # 从第一个操作数提取字符串
        operand_str = operands_node.children[0].value[0]

        # 解析寄存器大小
        left = operand_str.find('[')
        right = operand_str.find(']')
        if left == -1 or right == -1 or right <= left:
            raise ValueError(f"Invalid creg format: {operand_str}")

        creg_size = int(operand_str[left + 1:right])
        self.env._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.env.creg = self.env._initial_creg.copy()

    def execute_mov(self, node):
        # 获取操作数节点
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError(f"MOV instruction requires 2 operands: {node}")

        # 提取操作数字符串
        dest_operand = operands_node.children[0].value[0]
        src_operand = operands_node.children[1].value[0]

        # 解析目标操作数
        def parse_operand(op_str):
            if op_str.startswith('R['):
                return 'R', int(op_str[2:-1])  # 提取寄存器索引
            elif op_str in ['PC', 'LR']:
                return op_str, None  # 特殊寄存器
            elif op_str.isdigit() or '.' in op_str:
                return 'imm', float(op_str)  # 立即数
            else:
                raise ValueError(f"Invalid operand format: {op_str}")

        # 解析源和目标
        dest_type, dest_val = parse_operand(dest_operand)
        src_type, src_val = parse_operand(src_operand)

        # 执行MOV逻辑
        if dest_type == 'R' and src_type == 'R':
            self.env.registers[dest_val] = self.env.registers[src_val]
        elif dest_type == 'R' and src_type == 'imm':
            self.env.registers[dest_val] = src_val
        elif dest_type == 'R' and src_type == 'PC':
            self.env.registers[dest_val] = self.env.pc
        elif dest_type == 'R' and src_type == 'LR':
            self.env.registers[dest_val] = self.env.lr
        elif dest_type == 'PC' and src_type == 'R':
            self.env.pc = self.env.registers[src_val]
        elif dest_type == 'LR' and src_type == 'R':
            self.env.lr = self.env.registers[src_val]
        else:
            raise ValueError(f"Unsupported MOV combination: {dest_operand} <- {src_operand}")

    def execute_u3(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 4:
            raise ValueError("U3 requires 4 operands (theta, phi, lambda, qubit)")

        # 解析前三个参数（支持立即数或R寄存器）
        theta = self._parse_parameter(operands_node.children[0], 'R')
        phi = self._parse_parameter(operands_node.children[1], 'R')
        lambda_ = self._parse_parameter(operands_node.children[2], 'R')

        # 解析第四个参数为量子寄存器
        qubit = self._parse_register_index(operands_node.children[3].value[0], 'q')

        # 量子寄存器范围检查
        if qubit >= self.env.qreg_size:
            raise ValueError(f"U3 qubit index out of range: {qubit} (qreg_size={self.env.qreg_size})")

        # 构建量子门
        u3_gate = self._create_u3_gate(theta, phi, lambda_)
        self._apply_single_qubit_gate(u3_gate, qubit)

    def execute_cnot(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("CNOT requires control and target qubits")

        # 解析量子寄存器参数
        control = self._parse_register_index(operands_node.children[0].value[0], 'q')
        target = self._parse_register_index(operands_node.children[1].value[0], 'q')

        # 有效性检查
        if control == target:
            raise ValueError("CNOT control and target qubits cannot be the same")
        if control >= self.env.qreg_size or target >= self.env.qreg_size:
            raise ValueError(f"CNOT qubit index out of range (qreg_size={self.env.qreg_size})")

        # 构建并应用CNOT门
        self._apply_cnot_gate(control, target)

    def execute_gps(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("GPS requires delta and qubit parameters")

        # 解析delta参数（支持立即数或R寄存器）
        delta = self._parse_parameter(operands_node.children[0], 'R')

        # 验证第二个参数格式（q寄存器）
        self._parse_register_index(operands_node.children[1].value[0], 'q')

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
        value_str = operand_node.value[0]
        try:
            # 尝试解析为立即数
            return float(value_str)
        except ValueError:
            # 解析为寄存器值
            reg_index = self._parse_register_index(value_str, register_prefix)
            return self.env.registers[reg_index]

    @staticmethod
    def _create_u3_gate(theta, phi, lambda_):
        """创建U3门矩阵"""
        return np.array([
            [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lambda_)) * np.cos(theta / 2)]
        ])

    def _apply_single_qubit_gate(self, gate, qubit):
        """应用单量子比特门"""
        # 构建全系统矩阵
        full_gate = 1
        for q in reversed(range(self.env.qreg_size)):
            full_gate = np.kron(full_gate, gate if q == qubit else np.eye(2))

        # 根据模拟模式应用操作
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = full_gate @ self.env.quantum_state
        else:
            self.env.quantum_state = full_gate @ self.env.quantum_state @ full_gate.conj().T

    def _apply_cnot_gate(self, control, target):
        """CNOT门实现"""
        # 使用矩阵构建方式
        dim = 2 ** self.env.qreg_size
        cnot = np.eye(dim)
        for i in range(dim):
            if (i >> (self.env.qreg_size - 1 - control)) & 1:  # 高位在前
                j = i ^ (1 << (self.env.qreg_size - 1 - target))
                cnot[i, i] = 0
                cnot[j, i] = 1

        # 统一应用门操作
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = cnot @ self.env.quantum_state
        else:
            self.env.quantum_state = cnot @ self.env.quantum_state @ cnot.conj().T

    def execute_measure(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("MEASURE requires 2 operands (qubit, creg)")

        # 解析量子寄存器和经典寄存器
        qubit = self._parse_register_index(operands_node.children[0].value[0], 'q')
        creg_idx = self._parse_register_index(operands_node.children[1].value[0], 'c')

        # 寄存器范围校验
        if qubit >= self.env.qreg_size:
            raise ValueError(f"Qubit index out of range: {qubit} (qreg_size={self.env.qreg_size})")
        if creg_idx >= len(self.env.creg):
            raise ValueError(f"CReg index out of range: {creg_idx} (creg_size={len(self.env.creg)})")

        # 生成投影算子（修正量子位序问题）
        proj0, proj1 = self._create_measurement_projectors(qubit)

        # 执行测量
        result = self._perform_measurement(proj0, proj1)

        # 存储结果并应用噪声
        self.env.creg[creg_idx] = result
        if self.env.error_model:
            self.env.apply_classical_noise(creg_idx)  # 新增经典寄存器噪声

    def _create_measurement_projectors(self, qubit):
        """创建测量投影矩阵（兼容不同位序表示）"""
        # 调整量子位序为高位优先
        target_bit = self.env.qreg_size - 1 - qubit
        dim = 2 ** self.env.qreg_size

        proj0 = np.zeros((dim, dim))
        proj1 = np.zeros((dim, dim))

        for i in range(dim):
            if (i >> target_bit) & 1 == 0:
                proj0[i, i] = 1
            else:
                proj1[i, i] = 1
        return proj0, proj1

    def _perform_measurement(self, proj0, proj1):
        """执行测量核心逻辑"""
        if self.env.simulation_mode == 'statevector':
            prob0 = np.vdot(self.env.quantum_state, proj0 @ self.env.quantum_state).real
            if np.random.rand() < prob0:
                self.env.quantum_state = (proj0 @ self.env.quantum_state) / np.sqrt(prob0 + 1e-10)
                return 0
            else:
                self.env.quantum_state = (proj1 @ self.env.quantum_state) / np.sqrt(1 - prob0 + 1e-10)
                return 1
        else:
            # 密度矩阵模式
            prob0 = np.trace(proj0 @ self.env.quantum_state).real
            if np.random.rand() < prob0:
                self.env.quantum_state = (proj0 @ self.env.quantum_state @ proj0) / (prob0 + 1e-10)
                return 0
            else:
                self.env.quantum_state = (proj1 @ self.env.quantum_state @ proj1) / (1 - prob0 + 1e-10)
                return 1

    def execute_add(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        dest = self._parse_register_index(operands_node.children[0].value[0], 'R')

        # 解析源操作数（可以是寄存器或立即数）
        src1 = self._parse_operand(operands_node.children[1].value[0])
        src2 = self._parse_operand(operands_node.children[2].value[0])

        # 处理立即数
        val1 = src1 if isinstance(src1, (int, float)) else self.env.registers[src1]
        val2 = src2 if isinstance(src2, (int, float)) else self.env.registers[src2]

        self.env.registers[dest] = val1 + val2
        self._set_flags(self.env.registers[dest])

    def execute_mul(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("MUL instruction requires 3 operands")

        dest = self._parse_register_index(operands_node.children[0].value[0], 'R')
        src1 = self._parse_register_index(operands_node.children[1].value[0], 'R')
        src2 = self._parse_register_index(operands_node.children[2].value[0], 'R')

        result = self.env.registers[src1] * self.env.registers[src2]
        self.env.registers[dest] = result
        self._set_flags(result)

    def execute_div(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("DIV instruction requires 3 operands")

        dest = self._parse_register_index(operands_node.children[0].value[0], 'R')
        src1 = self._parse_register_index(operands_node.children[1].value[0], 'R')
        src2 = self._parse_register_index(operands_node.children[2].value[0], 'R')

        # 添加除零检查
        if self.env.registers[src2] == 0:
            raise ValueError("Division by zero")

        result = self.env.registers[src1] // self.env.registers[src2]
        self.env.registers[dest] = result
        self._set_flags(result)

    def execute_sub(self, node):
        # 获取操作数节点
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("SUB instruction requires 3 operands")

        # 解析目标寄存器
        dest_str = operands_node.children[0].value[0]
        dest = self._parse_register_index(dest_str, prefix='R')

        # 解析源寄存器1
        src1_str = operands_node.children[1].value[0]
        src1 = self._parse_register_index(src1_str, prefix='R')

        # 解析源寄存器2
        src2_str = operands_node.children[2].value[0]
        src2 = self._parse_register_index(src2_str, prefix='R')

        # 执行运算
        result = self.env.registers[src1] - self.env.registers[src2]
        self.env.registers[dest] = result
        self._set_flags(result)

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

    def _parse_operand(self, operand_str):
        try:
            # 尝试解析为数值
            return float(operand_str) if '.' in operand_str else int(operand_str)
        except ValueError:
            # 解析为寄存器
            prefix = 'R' if operand_str.startswith('R') else \
                'q' if operand_str.startswith('q') else 'c'
            return self._parse_register_index(operand_str, prefix)

        # 通用标志位设置方法
    def _set_flags(self, value):
        self.env.SF = 1 if value < 0 else 0
        self.env.ZF = 1 if value == 0 else 0

    def execute_b(self, node):
        label = self._parse_label_operand(node)
        self.env.pc = self._get_label_address(label)

    def execute_bl(self, node):
        label = self._parse_label_operand(node)
        self.env.lr = self.env.pc + 1  # 保存返回地址
        self.env.pc = self._get_label_address(label)

    def execute_beq(self, node):
        self.execute_conditional_branch(node, 'EQ')

    def execute_bne(self, node):
        self.execute_conditional_branch(node, 'NE')

    def execute_bgt(self, node):
        self.execute_conditional_branch(node, 'GT')

    def execute_bge(self, node):
        self.execute_conditional_branch(node, 'GE')

    def execute_blt(self, node):
        self.execute_conditional_branch(node, 'LT')

    def execute_ble(self, node):
        self.execute_conditional_branch(node, 'LE')

    def execute_conditional_branch(self, node, condition_type):
        """条件分支的统一处理"""
        if self._condition_met(condition_type):
            label = self._parse_label_operand(node)
            self.env.pc = self._get_label_address(label)
        else:
            self.env.pc += 1  # 自动递增PC
    # 辅助方法 ---------------------------------------------------
    @staticmethod
    def _parse_label_operand(node):
        """从分支指令节点解析标签操作数"""
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or not operands_node.children:
            raise ValueError(f"{node.children[0].value[0]} instruction missing label operand")

        label_operand = operands_node.children[0]
        if label_operand.type != 'Label':
            raise ValueError(f"Expected label operand, got {label_operand.type}")

        return label_operand.value[0].strip(':')  # 去除可能存在的冒号

    def _get_label_address(self, label):
        """获取标签地址并验证存在性"""
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
        dest_reg = self._parse_register_index(operands_node.children[0].value[0], 'R')

        # 解析源内存地址 M[...]
        src_mem = self._parse_memory_address(operands_node.children[1].value[0])

        # 执行加载操作
        self.env.registers[dest_reg] = self.env.memory[src_mem]

    def execute_str(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("STR requires 2 operands (src_reg, dest_mem)")

        # 解析源寄存器 R[...]
        src_reg = self._parse_register_index(operands_node.children[0].value[0], 'R')

        # 解析目标内存地址 M[...]
        dest_mem = self._parse_memory_address(operands_node.children[1].value[0])

        # 执行存储操作
        self.env.memory[dest_mem] = self.env.registers[src_reg]

    def execute_cldr(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 3:
            raise ValueError("CLDR requires 3 operands (dest_creg, real_mem, imag_mem)")

        # 解析目标经典寄存器 c[...]
        dest_creg = self._parse_register_index(operands_node.children[0].value[0], 'c')

        # 解析实部和虚部内存地址
        real_mem = self._parse_memory_address(operands_node.children[1].value[0])
        imag_mem = self._parse_memory_address(operands_node.children[2].value[0])

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
        src_creg = self._parse_register_index(operands_node.children[0].value[0], 'c')

        # 解析目标内存地址
        real_mem = self._parse_memory_address(operands_node.children[1].value[0])
        imag_mem = self._parse_memory_address(operands_node.children[2].value[0])

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

