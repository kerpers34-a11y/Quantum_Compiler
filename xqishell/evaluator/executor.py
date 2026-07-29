"""指令执行与程序求值(Evaluator)。

负责:预解析(shot/标签/指令体)、shot 主循环、全部 execute_* 指令实现、
分支/标志位/寄存器与内存操作。
由 evaluator.py 拆包而来(见 xqishell/evaluator/__init__.py)。
"""

import numpy as np

from xqishell import config
from xqishell.errors import XQIExecutionError

from .io import StateIO


class Evaluator(StateIO):
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

    def evaluate(self, ast, verbose=True):
        """执行完整程序:预解析 → 逐 shot 执行 → 最终输出。

        verbose: 是否打印 PC 轨迹等过程输出(默认 True,与历史行为一致)。
        """
        self._prepare(ast)
        for shot_nth in range(1, self.shot_total + 1):
            self._run_single_shot(shot_nth, verbose)
        self._report_final()

    def _prepare(self, ast):
        """预解析 (Shot, Labels, Instructions) 并初始化输出文件。"""
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

    def _run_single_shot(self, shot_nth, verbose=True):
        """执行单个 shot,结束后统计并追加二进制状态。"""
        if shot_nth == 1 and verbose:
            print(f"Total Program Row:{len(self.body_instructions):-10d}\n")

        self.shot_idx = shot_nth
        self.env.reset_for_shot()  # 内含 measured_bits 重置
        self.pre_measure_state_sv = None  # 每个 shot 重置，捕获第一次测量前的状态

        while self.env.pc < len(self.body_instructions) and not self.env.shot_completed:
            # 打印 PC 轨迹，严格匹配 C 输出：PC=%-10d (shot: %d)
            if verbose:
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
        if verbose:
            print("")

    def _report_final(self):
        """最终控制台输出(匹配 C 语言顺序)并收尾二进制文件。"""
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

    # ---------------------------------------------------------------- error 指令
    @staticmethod
    def _error_param(operands, idx, default):
        """取第 idx 个物理参数(缺省用默认值)。"""
        return float(operands[idx].value) if len(operands) > idx else default

    def _error_model_code1(self, ops):
        """去极化: enable, 1, p1, p2, p_measure, p_reset"""
        P = self._error_param
        return (1,
                P(ops, 2, config.DEFAULT_Q1_ERROR_PROBABILITY),
                P(ops, 3, config.DEFAULT_Q2_ERROR_PROBABILITY),
                P(ops, 4, config.DEFAULT_MEASURE_ERROR_PROBABILITY),
                P(ops, 5, config.DEFAULT_RESET_ERROR_PROBABILITY))

    def _error_model_code23(self, ops, code):
        """幅度(2)/相位(3)衰减: enable, code, gamma, p_measure, p_reset"""
        P = self._error_param
        default_gamma = (
            config.DEFAULT_AMP_DAMPING_GAMMA if code == 2
            else config.DEFAULT_PHASE_DAMPING_GAMMA
        )
        gamma = P(ops, 2, default_gamma)
        return (code, gamma, gamma,
                P(ops, 3, config.DEFAULT_MEASURE_ERROR_PROBABILITY),
                P(ops, 4, config.DEFAULT_RESET_ERROR_PROBABILITY))

    def _error_model_code4(self, ops):
        """热弛豫: enable, 4, T1, T2, Tgate, p_measure, p_reset"""
        P = self._error_param
        t = [P(ops, 2, config.DEFAULT_THERMAL_RELAXATION_ERROR_T1),
             P(ops, 3, config.DEFAULT_THERMAL_RELAXATION_ERROR_T2),
             P(ops, 4, config.DEFAULT_THERMAL_RELAXATION_ERROR_TGATE)]
        return (4, list(t), list(t),
                P(ops, 5, config.DEFAULT_MEASURE_ERROR_PROBABILITY),
                P(ops, 6, config.DEFAULT_RESET_ERROR_PROBABILITY))

    def _error_model_code5(self, ops):
        """Pauli: enable, 5, px, py, pz, p_measure, p_reset"""
        P = self._error_param
        p = [P(ops, 2, config.DEFAULT_PAULI_X_ERROR_PROBABILITY),
             P(ops, 3, config.DEFAULT_PAULI_Y_ERROR_PROBABILITY),
             P(ops, 4, config.DEFAULT_PAULI_Z_ERROR_PROBABILITY)]
        return (5, list(p), list(p),
                P(ops, 5, config.DEFAULT_MEASURE_ERROR_PROBABILITY),
                P(ops, 6, config.DEFAULT_RESET_ERROR_PROBABILITY))

    def _error_model_code6(self, ops):
        """相干幺正: enable, 6, ex, ey, ez, p_measure, p_reset"""
        P = self._error_param
        e = [P(ops, 2, config.DEFAULT_COHERENT_X_UNITARY_ERROR_PROBABILITY),
             P(ops, 3, config.DEFAULT_COHERENT_Y_UNITARY_ERROR_PROBABILITY),
             P(ops, 4, config.DEFAULT_COHERENT_Z_UNITARY_ERROR_PROBABILITY)]
        return (6, list(e), list(e),
                P(ops, 5, config.DEFAULT_MEASURE_ERROR_PROBABILITY),
                P(ops, 6, config.DEFAULT_RESET_ERROR_PROBABILITY))

    @staticmethod
    def _error_model_default(code):
        """未识别 code 的默认模型(沿用历史行为)。"""
        return (code, config.DEFAULT_Q1_ERROR_PROBABILITY,
                config.DEFAULT_Q2_ERROR_PROBABILITY,
                config.DEFAULT_MEASURE_ERROR_PROBABILITY,
                config.DEFAULT_RESET_ERROR_PROBABILITY)

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

        # 3. 处理第二个参数：Error Code,并按 code 分派构建误差模型
        code = config.DEFAULT_Q_ERROR_CODE if len(operands) < 2 else int(operands[1].value)

        builders = {
            1: self._error_model_code1,
            4: self._error_model_code4,
            5: self._error_model_code5,
            6: self._error_model_code6,
        }
        if code in (2, 3):
            self.env.error_model = self._error_model_code23(operands, code)
        elif code in builders:
            self.env.error_model = builders[code](operands)
        else:
            self.env.error_model = self._error_model_default(code)

        # 处理延迟应用逻辑
        if hasattr(self.env, '_pending_error_model'):
            self.env._pending_error_model = self.env.error_model

    @staticmethod
    def _reg_size_from(node):
        """从 qreg/creg 指令节点解析寄存器大小(q[n]/c[n] 中的 n)。"""
        operand_str = next((c for c in node.children if c.type == 'Operands'), None).children[0].value
        left, right = operand_str.find('['), operand_str.find(']')
        return int(operand_str[left + 1:right])

    def execute_qreg(self, node):
        qreg_size = self._reg_size_from(node)
        self.env.resize_qreg(qreg_size)
        print(f"Quantum Register Number: {qreg_size}")

    def execute_creg(self, node):
        creg_size = self._reg_size_from(node)
        self.env._initial_creg = np.zeros(creg_size, dtype=np.complex128)
        self.env.creg = self.env._initial_creg.copy()
        # 匹配 C 语言 613 行
        print(f"Classical Register Number: {creg_size}")
    # ---------------------------------------------------------------- MOV
    def _parse_mov_dest(self, dest_str):
        """解析 MOV 目标:返回 (类型, R索引或None)。"""
        if dest_str in {'PC', 'LR', 'SF', 'ZF'}:
            return dest_str, None
        if dest_str.startswith('R[') and dest_str.endswith(']'):
            try:
                idx = int(dest_str[2:-1])
            except ValueError:
                raise XQIExecutionError(f"Invalid R register format: {dest_str}") from None
            if idx < 0 or idx >= len(self.env.registers):
                raise XQIExecutionError(f"R[{idx}] out of range")
            return 'R', idx
        raise XQIExecutionError(f"Unsupported destination: {dest_str}")

    def _parse_mov_src(self, src_str):
        """解析 MOV 源:返回 (类型, R索引/立即数值或None)。

        支持立即数(整/浮)、R 寄存器、PC、LR、SF、ZF。
        """
        if src_str in {'PC', 'LR', 'SF', 'ZF'}:
            return src_str, None
        if src_str.startswith('R[') and src_str.endswith(']'):
            try:
                idx = int(src_str[2:-1])
            except ValueError:
                raise XQIExecutionError(f"Invalid source R register: {src_str}") from None
            if idx < 0 or idx >= len(self.env.registers):
                raise XQIExecutionError(f"Source R[{idx}] out of range")
            return 'R', idx
        if src_str.isdigit() or src_str.lstrip('-').isdigit():
            return 'imm', int(src_str)
        if '.' in src_str or 'e' in src_str.lower() or 'E' in src_str:
            return 'imm', float(src_str)
        raise XQIExecutionError(f"Unsupported source operand: {src_str}")

    def _src_value(self, src_type, src_val):
        """取源操作数的当前值。"""
        if src_type == 'R':
            return self.env.registers[src_val]
        if src_type == 'imm':
            return src_val
        if src_type == 'PC':
            return self.env.pc
        if src_type == 'LR':
            return self.env.lr
        if src_type == 'SF':
            return float(self.env.SF)
        if src_type == 'ZF':
            return float(self.env.ZF)
        raise XQIExecutionError(f"Unsupported source: {src_type}")

    def _mov_to_r(self, dest_val, src_type, src_val):
        value = self._src_value(src_type, src_val)
        self.env.registers[dest_val] = value
        # 写入普通 R 寄存器后，更新标志位（模仿算术指令）
        self._set_flags(value)

    def _mov_to_pc(self, _dest_val, src_type, src_val):
        if src_type == 'imm' and src_val == 0:
            # 特殊语义：结束当前 shot
            self.env.shot_completed = True
            return
        if src_type in {'SF', 'ZF'}:
            raise XQIExecutionError("Cannot MOV SF/ZF directly to PC")
        if src_type == 'R':
            self.env.pc = int(self.env.registers[src_val])
        elif src_type == 'imm':
            self.env.pc = int(src_val)
        elif src_type == 'LR':
            self.env.pc = self.env.lr
        else:
            raise XQIExecutionError(f"Unsupported src → PC: {src_type}")
        # PC 变更不更新标志位

    def _mov_to_lr(self, _dest_val, src_type, src_val):
        if src_type in {'SF', 'ZF'}:
            raise XQIExecutionError("Cannot MOV SF/ZF directly to LR")
        if src_type == 'R':
            self.env.lr = int(self.env.registers[src_val])
        elif src_type == 'imm':
            self.env.lr = int(src_val)
        elif src_type == 'PC':
            self.env.lr = self.env.pc
        else:
            raise XQIExecutionError(f"Unsupported src → LR: {src_type}")
        # LR 变更不更新标志位

    def _mov_to_sf(self, _dest_val, src_type, src_val):
        if src_type in {'PC', 'LR'}:
            raise XQIExecutionError("Cannot MOV PC/LR to SF")
        val = int(self._src_value(src_type, src_val))
        self.env.SF = 1 if val != 0 else 0 # 通常 SF=1 表示负数，这里简化处理为非零即1

    def _mov_to_zf(self, _dest_val, src_type, src_val):
        if src_type in {'PC', 'LR'}:
            raise XQIExecutionError("Cannot MOV PC/LR to ZF")
        val = int(self._src_value(src_type, src_val))
        self.env.ZF = 1 if val == 0 else 0

    def execute_mov(self, node):
        """
        执行 MOV 指令，支持：
        - 普通寄存器间 MOV
        - 立即数到寄存器
        - PC/LR ↔ 寄存器/立即数
        - 寄存器/立即数 → SF / ZF（会直接影响标志位）
        - 写入普通 R 寄存器后自动更新 SF 和 ZF
        """
        ops = self._require_operands(node, "MOV", 2, "dest, src")
        dest_type, dest_val = self._parse_mov_dest(ops[0].value.strip())
        src_type, src_val = self._parse_mov_src(ops[1].value.strip())
        # 目标类型二级分派(R / PC / LR / SF / ZF)
        handler = {
            'R': self._mov_to_r,
            'PC': self._mov_to_pc,
            'LR': self._mov_to_lr,
            'SF': self._mov_to_sf,
            'ZF': self._mov_to_zf,
        }[dest_type]
        handler(dest_val, src_type, src_val)

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

