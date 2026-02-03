import os
import time, math, cmath
import numpy as np
from xqi import config
from itertools import product

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
        elif node.type == 'Instruction' and node.children[0].value not in {'qreg', 'creg', 'error', 'XQI-BEGIN', 'XQI-END'}:
            pc += 1  # Only count body instructions
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
        self.error_model = None

    def _reset_quantum_register(self, new_size):
        """重置量子寄存器"""
        self.qreg_size = new_size
        self.initial_quantum_state = self._initialize_quantum_state(new_size)
        self.quantum_state = self.initial_quantum_state.copy()

    def resize_qreg(self, new_size):
        if not isinstance(new_size, int) or new_size < 0:
            raise ValueError("qreg_size must be non-negative integer")
        self._reset_quantum_register(new_size)
        if new_size > 0 and hasattr(self, '_pending_error_model'):
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
        np.random.seed(int(time.time() * 1000000) % 2 ** 32)
        self.quantum_state = self.initial_quantum_state.copy()
        self.creg = self._initial_creg.copy()
        self.registers.fill(0.0)  # 重置通用寄存器
        self.memory.fill(0.0)
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
        self.quantum_state = self.initial_quantum_state.copy()
        self.creg = self._initial_creg.copy()
        self.registers.fill(0.0)
        self.memory.fill(0.0)
        self.pc = 0
        self.lr = 0

    def apply_quantum_noise(self, qubits):
        """应用当前配置的量子噪声"""
        if self.qreg_size == 0 or not qubits:  # Early return for trivial cases
            return
        print(f"Applying noise to qubits {qubits} with model {self.error_model}")
        if not self.error_model and hasattr(self, '_pending_error_model'):
            self.error_model = self._pending_error_model
            print(f"Applying delayed error model at first noise application: {self._pending_error_model}")
            del self._pending_error_model
        if not self.error_model or self.error_model[0] == 9:
            return  # 测量错误单独处理
        noise_type, *params = self.error_model
        kraus_ops = self.generate_kraus_operators(noise_type, qubits, params)
        if any(not isinstance(k, np.ndarray) for k in kraus_ops):
            raise TypeError("Kraus operators must be numpy arrays")

        # 修改后的逻辑：统一采样Kraus，无论模式
        # 计算各Kraus的概率
        if self.simulation_mode == 'statevector':
            probs = np.array([np.linalg.norm(k @ self.quantum_state) ** 2 for k in kraus_ops])
        else:  # density_matrix
            probs = np.array([np.real(np.trace(k @ self.quantum_state @ k.conj().T)) for k in kraus_ops])
        probs /= np.sum(probs)  # 归一化
        # 随机选择并应用
        chosen = np.random.choice(len(kraus_ops), p=probs)
        if self.simulation_mode == 'statevector':
            new_state = kraus_ops[chosen] @ self.quantum_state
            norm = np.linalg.norm(new_state)
            self.quantum_state = new_state / norm if norm > 0 else new_state
        else:  # density_matrix
            new_rho = kraus_ops[chosen] @ self.quantum_state @ kraus_ops[chosen].conj().T
            trace = np.real(np.trace(new_rho))
            if trace > 0:
                new_rho /= trace
            else:
                print("Warning: trace after noise = 0, state unchanged")
            self.quantum_state = new_rho

        assert isinstance(self.quantum_state, np.ndarray), \
            f"quantum_state corrupted: {type(self.quantum_state)}"

    def generate_kraus_operators(self, noise_type, qubits, params):
        """生成对应噪声模型的Kraus算子"""
        operators = []
        # 确保 qubits 是一个列表（即使只有一个量子位）
        if not isinstance(qubits, list):
            qubits = [qubits]

        if self.qreg_size == 0:
            return [np.eye(1, dtype=np.complex128)]

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
                op_list = [np.eye(2) for _ in range(self.qreg_size)]
                for q, op in zip(qubits, combo):
                    op_list[q] = op
                full_op = np.eye(1, dtype=np.complex128)
                for q_idx in reversed(range(self.qreg_size)):
                    full_op = np.kron(full_op, op_list[q_idx])
                operators.append(full_op)

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
                # 构建全系统算子
                full_op = np.eye(1, dtype=np.complex128)
                for q_idx in reversed(range(self.qreg_size)):
                    full_op = np.kron(full_op, op_list[q_idx])
                operators.append(full_op)
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
                # 张量积构建
                full_op = np.eye(1, dtype=np.complex128)
                for q_idx in reversed(range(self.qreg_size)):
                    full_op = np.kron(full_op, op_list[q_idx])
                operators.append(full_op)
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
                full_op = np.eye(1, dtype=np.complex128)
                for q_idx in reversed(range(self.qreg_size)):
                    full_op = np.kron(full_op, op_list[q_idx])
                operators.append(full_op)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        self.validate_kraus_ops(operators)

        operators = [np.asarray(op, dtype=np.complex128) for op in operators]

        for i, op in enumerate(operators):
            if op.ndim != 2 or op.shape[0] != op.shape[1]:
                raise ValueError(f"Kraus operator {i} has invalid shape {op.shape}")
        self.validate_kraus_ops(operators)

        if noise_type == 1:
            print(f"generating {len(operators)} kraus ops for {len(qubits)} qubits")

        return operators

    @staticmethod
    def validate_kraus_ops(operators):
        """验证Σ Ki†Ki = I 的完备性关系"""

        if not operators:
            return  # Allow empty for trivial cases
        first_op = operators[0]
        dim = first_op.shape[0]
        if dim == 0:  # Skip validation for empty (though we now avoid this)
            return

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
    np.random.seed()
    def __init__(self, env, parser, ast):
        self.env = env
        self.labels = {}
        self.body_instructions = []
        self.source_code_text = parser.get_source_code_text(ast)
        self.labels_info = parser.get_labels_info(ast)
        self.parser = parser

    def evaluate(self, ast):
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

        # ==============================================
        # 第二阶段：按源代码顺序执行所有配置指令
        # （qreg / creg / error 执行）
        # ==============================================
        print("=== Executing configuration instructions in source order ===")
        config_opcodes = {'qreg', 'creg', 'error'}

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
            print(f"\n=== Starting shot {shot_idx + 1}/{shots} ===")

            # 每次 shot 重置量子态、经典寄存器、PC 等
            self.env.reset_for_shot()
            if self.env.error_model and self.env.error_model[0] != 9:  # Apply gate-like noise to all qubits
                self.env.apply_quantum_noise(list(range(self.env.qreg_size)))

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
            while self.env.pc < len(self.body_instructions):
                instr_node = self.body_instructions[self.env.pc]
                # node_id = id(instr_node)
                # if node_id in executed:
                #     print(f" WARNING: Skipping duplicate node (id {node_id})")
                #     self.env.pc += 1  # Still increment to avoid infinite loop
                #     continue
                # executed.add(node_id)
                opcode = next((c for c in instr_node.children if c.type == 'Opcode'),
                              None).value if instr_node.children else "unknown"

                print(f" Executing {opcode} at PC={self.env.pc}")
                self.execute_instruction(instr_node)

            results.append(self.env.creg.copy())
            print(f"  Shot {shot_idx + 1} completed. creg = {self.env.creg}")

        return results  # 如果需要返回所有 shots 的结果

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
            if opcode in ['CNOT', 'U3']:
                qubits = self._get_affected_qubits(node)
                if qubits:
                    print(f"Unified noise apply for {opcode}")
                    self.env.apply_quantum_noise(qubits)

            opcode = opcode_node.value
            special_mov_dests = {'PC', 'LR', 'SF', 'ZF'}
            branch_ops = {'B', 'BL', 'BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'}  # 移除 'MOV'
            if opcode in branch_ops:
                pass  # No auto +1
            elif opcode == 'MOV':
                dest_operand = operands_node.children[0].value if operands_node else None
                if dest_operand in special_mov_dests:
                    pass  # No +1 for MOV to PC/LR/etc.
                else:
                    self.env.pc += 1
            elif opcode in {'ADD', 'SUB', 'MUL', 'DIV', 'U3', 'CNOT', 'reset', 'measure', 'barrier', 'debug',
                            'debug-p'}:
                self.env.pc += 1

    @staticmethod
    def _get_affected_qubits(instr_node):  # 改名更清晰，接收的是 Instruction 节点
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

        if not operands_node or len(operands_node.children) < 2:
            # 如果只有一个操作数，则为第二个操作数设置默认值
            if len(operands_node.children) == 1:
                operands_node.children.append(ASTNode('Operand', value=config.default_Q_error_Code))  # 添加默认的第二个操作数

        # 检查是否有至少两个操作数
        if len(operands_node.children) < 1:
            raise ValueError("Invalid error instruction format")

        # 提取第一个操作数作为error_type
        error_type_operand = operands_node.children[0]
        error_type = int(error_type_operand.value)

        # 提取后续操作数作为参数
        params = [float(op.value) for op in operands_node.children[1:]]
        print(f"Depolarizing noise set with p = {params[0]}")


        # 设置错误模型
        if self.env.qreg_size > 0:
            self.env.error_model = (error_type, *params)
        else:
            self.env._pending_error_model = (error_type, *params)

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
            # 跳转到 PC
            if src_type == 'R':
                self.env.pc = int(self.env.registers[src_val])
            elif src_type == 'imm':
                self.env.pc = int(src_val)
            elif src_type == 'LR':
                self.env.pc = self.env.lr
            elif src_type in {'SF', 'ZF'}:
                raise ValueError("Cannot MOV SF/ZF directly to PC")
            else:
                raise ValueError(f"Unsupported src → PC: {src_type}")
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
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 4:
            raise ValueError("U3 requires 4 operands (theta, phi, lambda, qubit)")

        # 解析前三个参数（支持立即数或R寄存器）
        theta = self._parse_parameter(operands_node.children[0], 'R')
        phi = self._parse_parameter(operands_node.children[1], 'R')
        lambda_ = self._parse_parameter(operands_node.children[2], 'R')

        # 解析第四个参数为量子寄存器
        qubit = self._parse_register_index(operands_node.children[3].value, 'q')

        # 量子寄存器范围检查
        if qubit >= self.env.qreg_size:
            raise ValueError(f"U3 qubit index out of range: {qubit} (qreg_size={self.env.qreg_size})")

        # 构建量子门
        u3_gate = self.create_u3_gate(theta, phi, lambda_)
        self._apply_single_qubit_gate(u3_gate, qubit)

        print(f"Applying U3(theta={theta}, phi={phi}, lambda={lambda_}) to q[{qubit}]")
        print("U3 matrix:\n", u3_gate)

    def execute_cnot(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        if not operands_node or len(operands_node.children) < 2:
            raise ValueError("CNOT requires control and target qubits")
        control = self._parse_register_index(operands_node.children[0].value, 'q')  # 用户写的第一个 = 控制
        target = self._parse_register_index(operands_node.children[1].value, 'q')
        if control == target:
            raise ValueError("CNOT control and target qubits cannot be the same")
        if control >= self.env.qreg_size or target >= self.env.qreg_size:
            raise ValueError(f"CNOT qubit index out of range (qreg_size={self.env.qreg_size})")
        self._apply_cnot_gate(control, target)  # 只调用一次
        self.env.quantum_state = np.copy(self.env.quantum_state)

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

    @staticmethod
    def create_u3_gate(theta, phi, lambda_):
        ct = math.cos(theta / 2)
        st = math.sin(theta / 2)
        el = cmath.exp(1j * lambda_)
        ep = cmath.exp(1j * phi)
        return np.array([
            [ct, -el * st],
            [ep * st, ep * el * ct]
        ], dtype=np.complex128)

    def _apply_single_qubit_gate(self, gate, qubit):
        """应用单量子比特门"""
        # 构建全系统矩阵
        def _apply_single_qubit_gate(self, gate, qubit):
            full_gate = np.eye(1, dtype=np.complex128)
            for q in range(self.env.qreg_size):  # 改成正向：q[0] LSB 先 kron
                if q == qubit:
                    full_gate = np.kron(full_gate, gate)
                else:
                    full_gate = np.kron(full_gate, np.eye(2))

            if self.env.simulation_mode == 'statevector':
                self.env.quantum_state = full_gate @ self.env.quantum_state
            else:
                self.env.quantum_state = full_gate @ self.env.quantum_state @ full_gate.conj().T
        assert isinstance(self.env.quantum_state, np.ndarray), \
            f"quantum_state corrupted: {type(self.env.quantum_state)}"
        if self.env.simulation_mode == 'density_matrix':
            rho = self.env.quantum_state
        print("Applying U3 to qubit", qubit)
        print("Local U3 gate:\n", gate)
        full_gate = np.eye(1, dtype=np.complex128)
        for q in reversed(range(self.env.qreg_size)):
            if q == qubit:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        print("Full gate matrix diagonal:", full_gate.diagonal().real)
        print("Full gate shape:", full_gate.shape)
        # ... 原有应用代码
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = full_gate @ self.env.quantum_state
        else:
            self.env.quantum_state = full_gate @ self.env.quantum_state @ full_gate.conj().T

    def _apply_cnot_gate(self, control, target):
        """control 和 target 是量子位索引，q[0] 是最低位 (LSB)"""
        dim = 2 ** self.env.qreg_size
        cnot = np.eye(dim, dtype=np.complex128)

        for i in range(dim):
            # 提取 control 位的值
            ctrl_bit = (i >> control) & 1

            if ctrl_bit == 1:
                # 翻转 target 位
                flipped_i = i ^ (1 << target)
                # 把原始位置的振幅移动到翻转后的位置
                cnot[flipped_i, i] = 1.0
                cnot[i, i] = 0.0  # 清除原始对角
            # else: 保持 identity（已由 eye 设置）

        # 可选：打印验证
        print("CNOT control q[{}] → target q[{}]".format(control, target))
        print("CNOT matrix diagonal:", [cnot[k, k] for k in range(dim)])
        print("Non-zero off-diagonal count:", np.count_nonzero(cnot - np.eye(dim)))

        # 应用
        if self.env.simulation_mode == 'statevector':
            self.env.quantum_state = cnot @ self.env.quantum_state
        else:  # density_matrix
            self.env.quantum_state = cnot @ self.env.quantum_state @ cnot.conj().T

        # 调试用：打印应用后的对角（可选）
        if self.env.simulation_mode == 'density_matrix':
            diag = [np.real(self.env.quantum_state[i, i]) for i in range(dim)]
            print("After CNOT diagonal:", diag)

        self.env.quantum_state = self.env.quantum_state.copy()

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
            probs = [0.0, 0.0]
            for i, amp in enumerate(self.env.quantum_state):
                bit = (i >> qubit) & 1   # 改成低位索引
                probs[bit] += abs(amp) ** 2
            probs = np.array(probs)
            probs /= np.sum(probs)
            outcome = np.random.choice([0, 1], p=probs)
            # 塌缩态
            mask = [(i >> (self.env.qreg_size - 1 - qubit)) & 1 == outcome
                    for i in range(len(self.env.quantum_state))]
            self.env.quantum_state = np.where(mask, self.env.quantum_state, 0)
            norm = np.linalg.norm(self.env.quantum_state)
            if norm > 0:
                self.env.quantum_state /= norm
            if self.env.error_model and self.env.error_model[0] == 9:
                p_flip = self.env.error_model[1] if len(self.env.error_model) > 1 else 0.01  # Default flip prob
                if np.random.rand() < p_flip:
                    outcome = 1 - outcome  # Bit-flip error
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

            if self.env.error_model and self.env.error_model[0] == 9:
                p_flip = self.env.error_model[1] if len(self.env.error_model) > 1 else 0.01  # Default flip prob
                if np.random.rand() < p_flip:
                    outcome = 1 - outcome  # Bit-flip error

            self.env.creg[creg_idx] = outcome

    def _create_measurement_projectors(self, qubit):
        dim = 2 ** self.env.qreg_size
        proj0 = np.zeros((dim, dim), dtype=np.complex128)
        proj1 = np.zeros((dim, dim), dtype=np.complex128)

        for i in range(dim):
            # 假设 q[0] 是最低位 → 用最低位判断
            if (i >> qubit) & 1 == 0:
                proj0[i, i] = 1.0
            else:
                proj1[i, i] = 1.0

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
        """
        统一处理二元算术/逻辑指令，并更新标志位
        operation: lambda x,y: ... 或 operator.add / sub / mul / etc.
        """
        operands = node.children[1].children  # 假设 Operands 节点
        if len(operands) < 3:
            raise ValueError(f"{operation.__name__.upper()} needs 3 operands")

        dest = self._parse_register_index(operands[0].value, 'R')
        src1 = self._parse_operand(operands[1].value)
        src2 = self._parse_operand(operands[2].value)

        val1 = src1 if isinstance(src1, (int, float)) else self.env.registers[src1]
        val2 = src2 if isinstance(src2, (int, float)) else self.env.registers[src2]

        if operation == 'add':
            result = val1 + val2
        elif operation == 'sub':
            result = val1 - val2
        elif operation == 'mul':
            result = val1 * val2
        elif operation == 'div':
            if val2 == 0:
                raise ValueError("Division by zero")
            result = val1 / val2  # 或 // 取决于整数/浮点
        else:
            raise ValueError(f"Unknown operation: {operation}")

        self.env.registers[dest] = result
        self._set_flags(result)

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
        self.env.ZF = 1 if abs(value) < 1e-10 else 0

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
            raise ValueError(f"{node.children[0].value} instruction missing label operand")
        label_operand = operands_node.children[0]
        if label_operand.type != 'Label':
            raise ValueError(f"Expected label operand, got {label_operand.type}")
        return label_operand.value.strip(':')  # 去除可能存在的冒号

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
        self.print_debug_info(shot_id=1)

    def execute_debug_p(self, node):
        self.print_debug_info(shot_id=1)
        input("Press 'p' to continue: ")

    def execute_reset(self, node):
        operands_node = next((c for c in node.children if c.type == 'Operands'), None)
        qubit = None
        if operands_node and operands_node.children:
            qubit = self._parse_register_index(operands_node.children[0].value, 'q')
        # 调用环境的重置方法，确保根据仿真模式初始化
        self.env.reset()
        # 应用噪声
        if self.env.error_model and qubit is not None:
            self.env.apply_quantum_noise([qubit])
        print("After reset, rho[0,0]:", self.env.quantum_state[0, 0])
        print("rho[1,1]:", self.env.quantum_state[1, 1] if self.env.simulation_mode == 'density_matrix' else "N/A")

    def execute_barrier(self, node):
        # No-op in this simulation
        pass

    def execute_rand(self, node):
        dest = int(node.children[0].value[2:-1])
        seed = int(node.children[1].value[2:-1])
        np.random.seed(int(self.env.registers[seed]))
        self.env.registers[dest] = np.random.uniform(0, 1)

    def print_debug_info(self, shot_id=1):

        current_rho = self.env.quantum_state.copy()
        # 根据模式选择文件名
        if self.env.simulation_mode == "statevector":
            filename = config.filename_debug
        elif self.env.simulation_mode == "density_matrix":
            filename = config.filename_debug_Density_Matrix
        else:
            filename = "XQI-QC-unknown.txt"

        # 日志文件写到源文件所在目录
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()
        filepath = os.path.join(log_dir, filename)

        # 如果文件不存在，先写入固定开头
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.source_code_text.strip() + "\n\n")
                f.write("Label Number   Sequence   Label Symbol\n")
                for idx, (seq, symbol) in enumerate(self.labels_info):
                    f.write(f"Label   {idx}:      {seq:<8} {symbol}\n")
                f.write("\n")

        # 以追加模式写入调试信息
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")

            # 区分两种模式
            if self.env.simulation_mode == "density_matrix":
                f.write("XQI: current Density Matrix states:\n\n")
                matrix = current_rho  # ← 用拷贝
                rows, cols = matrix.shape
                f.write(f"matrix rows:{rows}, matrix columns:{cols}:\n")
                for i in range(rows):
                    for j in range(cols):
                        val = matrix[i, j]
                        f.write(f"[{i}][{j}]:({val.real:.6f})+({val.imag:.6f})i\n")
                f.write("\n\n")

            elif self.env.simulation_mode == "statevector":
                f.write("XQI: current states (High Qubit->Low Qubit) :\n")
                state = self.env.quantum_state
                dim = len(state)
                # 输出态矢量
                for idx, amp in enumerate(state):
                    bin_str = format(idx, f"0{int(np.log2(dim))}b")
                    f.write(f"state {bin_str}:  ({amp.real:.6f})+({amp.imag:.6f})i\n")
                f.write("corresponding Density Matrix state:\n\n")

                # 转换为密度矩阵
                rho = np.outer(state, np.conjugate(state))
                rows, cols = rho.shape
                f.write(f"matrix rows:{rows}, matrix columns:{cols}:\n")
                for i in range(rows):
                    for j in range(cols):
                        val = rho[i, j]
                        f.write(f"[{i}][{j}]:({val.real:.6f})+({val.imag:.6f})i\n")
                f.write("\n\n")

                # 输出概率分布
                f.write("current states probability (High Qubit->Low Qubit) :\n")
                probs = np.abs(state) ** 2
                for idx, p in enumerate(probs):
                    bin_str = format(idx, f"0{int(np.log2(dim))}b")
                    f.write(f"state {bin_str}:    {p:.6f}\n")
                f.write("\n\n")

            # 打印寄存器
            f.write(" register:\n")
            for idx, val in enumerate(self.env.registers):
                f.write(f"R[{idx:2d}]=   {val:.10f}\n")
            f.write("\n")

            # 打印 CPSR
            f.write(f"CPSR: SIGN_FLAG={self.env.SF};  ZERO_FLAG={self.env.ZF}.\n\n")

            # 打印内存
            f.write(" memory:\n")
            for idx, val in enumerate(self.env.memory):
                f.write(f"M[{idx:4d}]=   {val:.10f}\n")
            f.write("\n\n")