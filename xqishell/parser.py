from xqishell import config
class ASTNode:
    """ 抽象语法树的通用节点 """
    def __init__(self, type_, value=None, children=None, line=None, col=None):
        self.type = type_
        self.value = value
        self.children = children or []
        self.line = line
        self.col = col

    def __repr__(self):
        children_repr = [str(child) if isinstance(child, ASTNode) else repr(child)
                        for child in self.children]
        return f"ASTNode('{self.type}', value={self.value}, children=[{', '.join(children_repr)}])"

class Parser:
    def __init__(self, lexer, source_path=None):
        self.lexer = lexer
        self.source_path = source_path
        self.current_token = self.lexer.next_token()
        self.token_map = lexer.token_map  # 保存 token_map

        self.last_valid_opcode = None
        self.last_opcode_line = -1

    def eat(self, token_type):
        """ 消耗当前 token，并获取下一个 token """
        if self.current_token[0] == token_type:
            token = self.current_token
            self.current_token = self.lexer.next_token()
            return token
        else:
            raise SyntaxError(
                f"语法错误: 期望 {token_type}, 但得到 {self.current_token[1]} (行 {self.current_token[2]}, 列 {self.current_token[3]})")

    def program(self):
        """ 解析完整的程序，支持顺序执行、分支、循环 """
        program_node = ASTNode("Program", line=self.current_token[2], col=self.current_token[3])

        # 确保 XQI-BEGIN 只出现在第一行
        if self.current_token[0] == 'XQI_BEGIN':
            program_node.children.append(self.xqi_begin())
        else:
            raise SyntaxError(f"程序必须以 XQI-BEGIN 开始 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

        # 标记是否找到 shot 和 error
        found_shot = False
        found_error = False

        # 跳过空行，并检查 shot 和 error
        while self.current_token[0] != 'EOF' and self.current_token[0] != 'XQI_END':
            # 跳过空行
            if self.current_token[0] == 'NEWLINE':
                self.eat('NEWLINE')
                continue

            # 检查 shot 和 error，允许顺序无关
            if self.current_token[0] == 'OPCODE':
                if self.current_token[1] == 'shot' and not found_shot:
                    # 记录 shot
                    program_node.children.append(self.instruction())
                    found_shot = True
                elif self.current_token[1] == 'error' and not found_error:
                    # 记录 error
                    program_node.children.append(self.instruction())
                    found_error = True
                else:
                    # 如果是其他操作码，继续解析
                    program_node.children.append(self.instruction())

            # 如果都找到了 shot 和 error，就可以跳出
            if found_shot and found_error:
                break

        # 如果没有找到 shot 或 error，抛出错误
        if not found_shot:
            raise SyntaxError(f"程序缺少 'shot' 指令 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
        if not found_error:
            raise SyntaxError(f"程序缺少 'error' 指令 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

        # 继续解析其他语句
        while self.current_token[0] != 'EOF' and self.current_token[0] != 'XQI_END':
            stmt = self.statement()
            if stmt:
                # 更新最后有效指令状态
                if stmt.type == "Instruction":
                    op = stmt.children[0].value
                    if op in ('U3', 'CNOT'):
                        self.last_valid_opcode = op
                        self.last_opcode_line = stmt.line
                else:
                    if stmt.type == "Instruction":
                        op = stmt.children[0].value
                        if op not in ('U3', 'CNOT') and stmt.line > self.last_opcode_line:
                            self.last_valid_opcode = None
                program_node.children.append(stmt)

        # 确保 XQI-END 只出现在最后一行
        if self.current_token[0] == 'XQI_END':
            program_node.children.append(self.xqi_end())  # 调用 xqi_end 方法
        else:
            raise SyntaxError(f"程序必须以 XQI-END 结束 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

        return program_node

    def xqi_begin(self):
        opcode_node = ASTNode("Opcode", 'XQI-BEGIN', line=self.current_token[2], col=self.current_token[3])
        self.eat('XQI_BEGIN')
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')
        return ASTNode("Instruction", children=[opcode_node], line=opcode_node.line, col=opcode_node.col)

    def xqi_end(self):
        opcode_node = ASTNode("Opcode", 'XQI-END', line=self.current_token[2], col=self.current_token[3])
        self.eat('XQI_END')
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')
        return ASTNode("Instruction", children=[opcode_node], line=opcode_node.line, col=opcode_node.col)

    def statement(self):
        """ 解析语句（指令、分支、过程调用、过程定义）"""
        if self.current_token[0] == 'LABEL_DEF':
            return self.process_definition()
        elif self.current_token[0] == 'OPCODE':
            return self.instruction()
        elif self.current_token[0] in ('B', 'BL'):
            return self.unconditional_branch()
        elif self.current_token[0] in ('BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'):
            return self.conditional_branch()
        elif self.current_token[0] == 'XQI_END':  # 直接返回，不进入 statement() 解析
            return None
        elif self.current_token[0] == 'ARROW':  # 发现孤立的 `->` 说明前面解析错误
            raise SyntaxError(
                f"孤立的 '->' 语法错误，可能缺少 `measure` 指令 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
        elif self.current_token[0] == 'ASSIGN':  # 处理单个的分号
            # 检查前一个 token 是否与当前 `ASSIGN` 在同一行
            if self.lexer.tokens and self.lexer.tokens[0][0] == 'EOF':
                # 如果下一个 token 是 EOF，直接消耗当前的 `ASSIGN` 并返回 None
                self.eat('ASSIGN')
                return None
            elif self.lexer.tokens:
                previous_token = self.lexer.tokens[0]
                if previous_token[2] == self.current_token[2]:  # 检查行号是否相同
                    # 如果行号相同，说明这个 `ASSIGN` 是某个语句的一部分，继续解析
                    self.eat('ASSIGN')
                    return None
                else:
                    # 如果行号不同，说明这个 `ASSIGN` 是单独的分号，消耗当前的 `ASSIGN` 并返回 None
                    self.eat('ASSIGN')
                    return None
            else:
                raise SyntaxError(
                    f"无效的语句: {self.current_token[1]} (行 {self.current_token[2]}, 列 {self.current_token[3]})")
        elif self.current_token[0] == 'EOF':  # 处理 EOF
            return None
        else:
            raise SyntaxError(
                f"无效的语句: {self.current_token[1]} (行 {self.current_token[2]}, 列 {self.current_token[3]})")

    def process_definition(self):
        """ 解析过程定义 (label:) """
        label_name = self.current_token[1][:-1]  # 移除冒号
        node = ASTNode("Label", label_name, line=self.current_token[2], col=self.current_token[3])
        self.eat('LABEL_DEF')
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')
        return node

    def instruction(self):
        """ 解析指令，确保 measure 作为独立的 Instruction """
        node = ASTNode("Instruction", line=self.current_token[2], col=self.current_token[3])
        opcode_node = self.opcode()

        if opcode_node.value == "ERR":
            return self.handle_err_instruction(self, node, opcode_node)
        if opcode_node.value == "GPS":
            operands_node = self.operand_list()
            self.validate_gps_operands(operands_node.children)
            node.children.extend([opcode_node, operands_node])
            self.eat('ASSIGN')
            return node

        if opcode_node.value == "measure":
            # 手动解析 measure 的操作数结构: q[X] -> c[Y]
            source = self.operand()  # 解析量子寄存器

            # 检查并消耗 ARROW
            if self.current_token[0] != 'ARROW':
                raise SyntaxError(f"measure指令缺少 -> (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            self.eat('ARROW')

            dest = self.operand()  # 解析经典寄存器

            # 构建操作数节点（仅包含源和目标）
            operands_node = ASTNode("Operands", children=[source, dest], line=source.line, col=source.col)
            self.validate_measure_operands(operands_node.children)
            node.children = [opcode_node, operands_node]

            # 确保后续没有多余的操作数或逗号
            while self.current_token[0] != 'ASSIGN' and self.current_token[0] != 'EOF':
                raise SyntaxError(f"measure指令参数过多 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

            self.eat('ASSIGN')
            return node
        elif opcode_node.value == "shot":
            # 解析 shot 操作数
            operands_node = self.operand_list()
            if len(operands_node.children) != 1 or not operands_node.children[0].value.isdigit() or int(operands_node.children[0].value) <= 0:
                raise SyntaxError(
                    f"语法错误: shot 后面必须且仅能接一个正整数 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            node.children = [opcode_node, operands_node]
        elif opcode_node.value == "error":
            operands_node = self.operand_list()
            if not self.validate_error_operands(operands_node.children):
                # validate 函数内部已经 raise，这里不会执行到
                pass
            node.children = [opcode_node, operands_node]
        elif opcode_node.value == "U3":
            operands_node = self.operand_list()
            self.validate_u3_operands(operands_node.children)
            node.children = [opcode_node, operands_node]
        elif opcode_node.value == "CNOT":
            operands_node = self.operand_list()
            self.validate_cnot_operands(operands_node.children)
            node.children = [opcode_node, operands_node]
        elif opcode_node.value in ["CLDR", "CSTR"]:
            operands_node = self.operand_list()
            self.validate_cldr_cstr_operands(opcode_node.value, operands_node.children)
            node.children = [opcode_node, operands_node]
        elif opcode_node.value in ["ADD", "SUB", "MUL", "DIV"]:  # 经典指令
            operands_node = self.operand_list()
            self.validate_classical_arithmetic_operands(operands_node.children)
            node.children = [opcode_node, operands_node]
        elif opcode_node.value == "MOV":
            operands_node = self.operand_list()
            self.validate_mov_operands(operands_node.children)
            node.children = [opcode_node, operands_node]
        else:
            # 其他指令的标准解析方式
            operands_node = self.operand_list()
            node.children = [opcode_node, operands_node]
        # 确保 `;` 结束
        self.eat('ASSIGN')

        return node

    @staticmethod
    def handle_err_instruction(self, node, opcode_node):
        """ 增强版 ERR 处理：支持多参数物理模型 """
        if self.last_valid_opcode not in ('U3', 'CNOT'):
            raise SyntaxError(f"ERR 必须紧跟在 U3 或 CNOT 之后 (行 {opcode_node.line})")

        operands_node = self.operand_list()
        all_ops = operands_node.children

        if len(all_ops) < 3:  # 最简格式: ERR(model, code) q[n]
            raise SyntaxError(f"ERR 指令参数不足 (行 {opcode_node.line})")

        # 1. 验证模型 (model)
        model_node = all_ops[0]
        self.validate_err_model(model_node)
        model_type = int(model_node.value)

        # 2. 找到第一个量子寄存器 q[...] 的位置，以此区分括号参数和目标比特
        qreg_start_idx = -1
        for idx, op in enumerate(all_ops):
            if op.value.startswith('q['):
                qreg_start_idx = idx
                break

        if qreg_start_idx == -1:
            raise SyntaxError(f"ERR 指令缺少目标量子寄存器 q[...] (行 {opcode_node.line})")

        # 3. 校验物理参数部分 (model, code 之后的参数)
        # 如果是 4, 5, 6 模型，通常后面跟着 3 个物理参数
        param_count = qreg_start_idx - 2  # 减去 model 和 code
        if model_type in (4, 5, 6) and param_count < 3:
            raise SyntaxError(f"模型 {model_type} 需要 3 个物理参数 (行 {opcode_node.line})")

        # 验证所有 q[...] 范围
        for qreg in all_ops[qreg_start_idx:]:
            self.validate_qubit_register(qreg)

        node.children = [
            opcode_node,
            ASTNode("Operands", children=all_ops, line=operands_node.line, col=operands_node.col)
        ]

        self.eat('ASSIGN')
        self.last_valid_opcode = 'ERR'
        self.last_opcode_line = opcode_node.line
        return node

    # 辅助验证方法
    @staticmethod
    def validate_err_model(node):
        """ 允许 1-6 种误差模型 """
        try:
            model_val = int(node.value)
            if model_val not in {1, 2, 3, 4, 5, 6}:
                raise ValueError
        except ValueError:
            raise SyntaxError(f"错误模型值必须为 1-6 之间的整数 (当前:{node.value} 行 {node.line})")

    @staticmethod
    def validate_err_code(node):
        if not node.value.isdigit() or not (0 <= int(node.value) <= 9):
            raise SyntaxError(f"错误代码必须0-9 (当前:{node.value} 行 {node.line})")

    @staticmethod
    def validate_err_probability(node):
        try:
            prob = float(node.value)
            if not 0.0 <= prob <= 1.0:
                raise ValueError
        except ValueError:
            raise SyntaxError(f"概率值必须0.0-1.0 (当前:{node.value} 行 {node.line})")

    def opcode(self):
        """ 解析操作码 """
        token = self.current_token
        if token[0] == 'OPCODE':
            node = ASTNode("Opcode", token[1], line=token[2], col=token[3])
            self.eat('OPCODE')
            return node
        else:
            raise SyntaxError(f"语法错误: 期望操作码，但得到 {token[0]} (行 {token[2]}, 列 {token[3]})")

    def operand_list(self):
        """ 解析操作数列表，生成扁平化结构 """
        operands_node = ASTNode("Operands", line=self.current_token[2], col=self.current_token[3])

        # 处理括号参数（直接展平）
        if self.current_token[0] == 'LPAREN':
            self.eat('LPAREN')
            while self.current_token[0] not in ('RPAREN', 'EOF'):
                # 解析单个操作数（支持寄存器/立即数/数字）
                operand = self.operand()
                operands_node.children.append(operand)

                # 处理逗号分隔符
                if self.current_token[0] == 'COMMA':
                    self.eat('COMMA')
                elif self.current_token[0] not in ('RPAREN', 'EOF'):
                    raise SyntaxError(
                        f"缺少逗号分隔符 (行 {self.current_token[2]}, 列 {self.current_token[3]})"
                    )

            if self.current_token[0] != 'RPAREN':
                raise SyntaxError(f"未闭合的括号 (行 {operands_node.line})")
            self.eat('RPAREN')

        # 处理主操作数（量子寄存器/经典寄存器等）
        while self.current_token[0] in ('REGISTER', 'REGISTER_M', 'REGISTER_C',
                                        'REGISTER_Q', 'IMMEDIATE', 'NUMBER', 'LABEL', 'REGISTER_LR', 'REGISTER_PC'):
            operands_node.children.append(self.operand())

            # 允许逗号分隔的非括号参数（如CNOT q[0], q[1]）
            if self.current_token[0] == 'COMMA':
                self.eat('COMMA')

        # 处理measure的特殊语法（->）
        if self.current_token[0] == 'ARROW':
            self.eat('ARROW')
            operands_node.children.append(self.operand())

        return operands_node

    def operand(self):
        """ 解析单个操作数 """
        token = self.current_token
        if token[0] == 'REGISTER':
            # 提取 R[n] 格式，并检查 n 是否在有效范围内
            register_number = int(token[1][2:-1])  # 提取并转换为整数
            if register_number < 0 or register_number >= config.MAX_Register:
                raise SyntaxError(f"语法错误: R[{register_number}] 超出范围 (行 {token[2]}, 列 {token[3]})")
            node = ASTNode("Operand", f"R[{register_number}]", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_M':
            # 提取 M[n] 格式，并检查 n 是否在有效范围内
            register_number = int(token[1][2:-1])
            if register_number < 0 or register_number >= config.MAX_Memory:
                raise SyntaxError(f"语法错误: M[{register_number}] 超出范围 (行 {token[2]}, 列 {token[3]})")
            node = ASTNode("Operand", f"M[{register_number}]", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_C':
            # 提取 c[n] 格式，并检查 n 是否在有效范围内
            register_number = int(token[1][2:-1])
            if register_number < 0 or register_number >= config.MAX_Classical_Register:
                raise SyntaxError(f"语法错误: c[{register_number}] 超出范围 (行 {token[2]}, 列 {token[3]})")
            node = ASTNode("Operand", f"c[{register_number}]", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_Q':
            # 提取 q[n] 格式，并检查 n 是否在有效范围内
            register_number = int(token[1][2:-1])
            if register_number < 0 or register_number >= config.MAX_QUBITS:
                raise SyntaxError(f"语法错误: q[{register_number}] 超出范围 (行 {token[2]}, 列 {token[3]})")
            node = ASTNode("Operand", f"q[{register_number}]", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_LR':
            node = ASTNode("Operand", "LR", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_PC':
            node = ASTNode("Operand", "PC", line=token[2], col=token[3])
        elif token[0] == 'COMPLEX':
            node = ASTNode("Operand", token[1], line=token[2], col=token[3])  # 复数
        elif token[0] == 'IMMEDIATE':
            node = ASTNode("Operand", token[1], line=token[2], col=token[3])  # 立即数
        elif token[0] == 'NUMBER':
            node = ASTNode("Operand", token[1], line=token[2], col=token[3])  # 数字
        elif token[0] == 'LABEL':
            node = ASTNode("Label", token[1], line=token[2], col=token[3])  # 标签
        else:
            raise SyntaxError(f"语法错误: 未知的操作数类型 {token[0]} (行 {token[2]}, 列 {token[3]})")
        self.eat(token[0])
        return node

    def unconditional_branch(self):
        opcode_token = self.current_token
        opcode_node = ASTNode("Opcode", opcode_token[1], line=opcode_token[2], col=opcode_token[3])
        self.eat(opcode_token[0])
        label_token = self.current_token
        label_node = ASTNode("Label", label_token[1], line=label_token[2], col=label_token[3])
        self.eat('LABEL')
        operands_node = ASTNode("Operands", children=[label_node], line=label_token[2], col=label_token[3])
        self.eat('ASSIGN')
        return ASTNode("Instruction", children=[opcode_node, operands_node], line=opcode_token[2], col=opcode_token[3])

    def conditional_branch(self):
        opcode_token = self.current_token
        opcode_node = ASTNode("Opcode", opcode_token[1], line=opcode_token[2], col=opcode_token[3])
        self.eat(opcode_token[0])
        label_token = self.current_token
        label_node = ASTNode("Label", label_token[1], line=label_token[2], col=label_token[3])
        self.eat('LABEL')
        operands_node = ASTNode("Operands", children=[label_node], line=label_token[2], col=label_token[3])
        self.eat('ASSIGN')
        return ASTNode("Instruction", children=[opcode_node, operands_node], line=opcode_token[2], col=opcode_token[3])

    @staticmethod
    def validate_mov_operands(operands):
        for op in operands:
            if isinstance(op, ASTNode) and op.type == "Operand":
                if op.value.startswith(('q[', 'c[', 'M[')):
                    raise SyntaxError(f"MOV指令禁止使用 {op.value} 寄存器 (行 {op.line})")

    @staticmethod
    def validate_measure_operands(operands):
        """ 验证 measure 指令操作数（已确保 -> 存在，AST 不包含它）"""
        if len(operands) != 2:
            raise SyntaxError(f"measure指令需要两个操作数，当前数量 {len(operands)}")

        q_register, c_register = operands[0], operands[1]

        if not q_register.value.startswith('q['):
            raise SyntaxError(f"measure源操作数必须是量子寄存器 (行 {q_register.line})")
        if not c_register.value.startswith('c['):
            raise SyntaxError(f"measure目标操作数必须是经典寄存器 (行 {c_register.line})")

    def validate_u3_operands(self, operands):
        # 参数数量验证
        if len(operands) != 4:
            raise SyntaxError(
                f"U3指令需要3个参数和1个量子寄存器，当前参数数 {len(operands)} (行 {self.current_token[2]})")

        # 参数类型验证（前三个参数）
        valid_param_types = {
            'number': lambda n: self.is_valid_number(n),
            'immediate': lambda n: n.type == "IMMEDIATE",
            'register': lambda n: any(n.value.startswith(prefix)
                                      for prefix in ('R[', 'c[', 'M['))
        }

        for idx, param in enumerate(operands[:3]):
            if not any(check(param) for check in valid_param_types.values()):
                raise SyntaxError(
                    f"U3参数{idx + 1}类型错误: {param.value}\n"
                    f"允许类型: 数字/立即数/经典寄存器 (行 {param.line})"
                )

            # 如果是寄存器，验证范围
            if valid_param_types['register'](param):
                self.validate_register_range(param)

        # 量子寄存器验证
        q_register = operands[3]
        if not q_register.value.startswith('q['):
            raise SyntaxError(f"U3目标必须是量子寄存器 (行 {q_register.line})")
        self.validate_qubit_register(q_register)

    @staticmethod
    def validate_error_operands(operands):
        """
        支持 Code 1, 2, 3, 4, 5, 6
        """
        if len(operands) < 1:
            raise SyntaxError("error 指令至少需要一个参数：TRUE 或 FALSE")

        # 增加上限到 7 (enable, code, p1, p2, p3, p_measure, p_reset)
        if len(operands) > 7:
            raise SyntaxError(f"error 指令最多允许 7 个参数，当前收到 {len(operands)} 个")

        enable_node = operands[0]
        enable_val = enable_node.value.strip().upper()

        if enable_val not in ('TRUE', 'FALSE', '1', '0'):
            raise SyntaxError(f"error 第一个参数必须是 TRUE/FALSE/1/0")

        if enable_val in ('FALSE', '0'):
            return True

        # 校验 Code 和后续参数
        if len(operands) > 1:
            code_node = operands[1]
            try:
                code = int(code_node.value)
            except ValueError:
                raise SyntaxError(f"error code 必须是整数 (行 {code_node.line})")

            # 统一校验所有概率/物理参数是否为有效数字
            for i, op in enumerate(operands[2:], start=3):
                try:
                    float(op.value)
                except ValueError:
                    raise SyntaxError(f"error 第 {i} 个参数必须是数字 (行 {op.line})")

        return True

    def validate_gps_operands(self, operands):
        """ 验证GPS指令格式：GPS(delta) q[m]; """
        if len(operands) != 2:
            raise SyntaxError(
                f"GPS指令需要2个操作数，当前数量 {len(operands)} (行 {self.current_token[2]})"
            )
        # 提取两个操作数
        delta_operand, q_operand = operands[0], operands[1]
        # 验证参数部分

        if q_operand.value.startswith('q['):
            raise SyntaxError(f"GPS参数delta不能是量子寄存器 (行 {q_operand.line})")

        # 验证操作数部分
        if len(delta_operand) != 1 or not delta_operand[0].value.startswith('q['):
            raise SyntaxError(f"GPS第二个操作数必须是量子寄存器 (行 {self.current_token[2]})")

        # 验证寄存器范围
        self.validate_qubit_register(q_operand)

    def validate_cnot_operands(self, operands):
        """ 验证CNOT指令操作数结构 """
        # 验证操作数数量
        if len(operands) != 2:
            raise SyntaxError(
                f"CNOT需要2个量子寄存器，检测到 {len(operands)} 个操作数\n"
                f"错误位置：行 {self.current_token[2]} 列 {self.current_token[3]}"
            )

        # 类型和格式验证
        for idx, op in enumerate(operands, 1):
            if not isinstance(op, ASTNode) or not op.value.startswith('q['):
                raise SyntaxError(
                    f"操作数 {idx} 类型错误: 期望量子寄存器，实际得到 {op.value}\n"
                    f"错误位置：行 {op.line} 列 {op.col}"
                )

        q_operand_0, q_operand_1 = operands[0], operands[1]
        self.validate_qubit_register(q_operand_0)
        self.validate_qubit_register(q_operand_1)

        control_qubit = int(operands[0].value[2:-1])
        target_qubit = int(operands[1].value[2:-1])

        # 验证控制位和目标位不同
        if control_qubit == target_qubit:
            raise SyntaxError(
                f"CNOT控制位和目标位不能相同\n"
                f"冲突寄存器: q[{control_qubit}]\n"
                f"错误位置：行 {operands[0].line} 和 行 {operands[1].line}"
            )

    def get_source_code_text(self, ast):
        """
        从 AST 生成带行号的源代码文本，仿照 Operate Instructions 格式
        """
        lines = ["Operate Instructions:"]
        for child in ast.children:
            if child.type == "Instruction":
                opcode = child.children[0].value
                operands = ""
                if len(child.children) > 1 and child.children[1].type == "Operands":
                    # 拼接操作数
                    operands = " ".join(op.value for op in child.children[1].children)
                # 行号 + 指令 + 操作数 + ;
                lines.append(f"     {child.line}: {opcode} {operands};")
            elif child.type == "Label":
                # Label 定义行号和符号
                lines.append(f"     {child.line}: {child.value}:")
        return "\n".join(lines)

    def get_labels_info(self, ast):
        """
        从 AST 收集所有 Label 节点，返回 (行号, 标签名) 列表
        """
        labels = []
        for child in ast.children:
            if child.type == "Label":
                labels.append((child.line, child.value))
        return labels

    @staticmethod
    def is_valid_number(node):
        """ 验证是否为有效数字（整数或浮点数） """
        if node.type != "Operand":
            return False
        try:
            float(node.value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_register_range(node):
        """ 验证经典寄存器范围 """
        value = node.value
        if value.startswith('R['):
            max_reg = config.MAX_Register
        elif value.startswith('c['):
            max_reg = config.MAX_Classical_Register
        elif value.startswith('M['):
            max_reg = config.MAX_Memory
        else:
            return  # 非寄存器类型不处理

        try:
            reg_num = int(value[2:-1])
            if not (0 <= reg_num < max_reg):
                raise ValueError
        except ValueError:
            raise SyntaxError(
                f"寄存器越界: {value} (允许范围 0-{max_reg - 1}) (行 {node.line})"
            )

    @staticmethod
    def validate_qubit_register(node):
        """ 验证量子寄存器范围 """
        try:
            reg_num = int(node.value[2:-1])
            if not (0 <= reg_num < config.MAX_QUBITS):
                raise ValueError
        except ValueError:
            raise SyntaxError(
                f"量子寄存器越界: {node.value} (允许范围 0-{config.MAX_QUBITS - 1}) (行 {node.line})"
            )

    @staticmethod
    def validate_classical_arithmetic_operands(operands):
        for op in operands:
            if isinstance(op, ASTNode) and op.type == "Operand" and op.value.startswith('q['):
                raise SyntaxError(f"经典算术指令禁止使用量子寄存器 (行 {op.line})")

    @staticmethod
    def validate_cldr_cstr_operands(opcode, operands):
        if len(operands) < 2:
            raise SyntaxError(f"{opcode}需要两个操作数")
        first, second = operands[0], operands[1]
        if opcode == "CLDR":
            if not first.value.startswith('c[') or not second.value.startswith('M['):
                raise SyntaxError(f"CLDR格式应为c[X], M[Y] (行 {first.line})")
        elif opcode == "CSTR":
            if not first.value.startswith('c[') or not second.value.startswith('M['):
                raise SyntaxError(f"CSTR格式应为c[X], M[Y] (行 {first.line})")

"""
    def loop(self):
        ### 解析循环结构
        start_label = None
        end_label = None
        # 寻找循环的开始标签
        if self.current_token[0] == 'LABEL_DEF':
            start_label = self.current_token[1][:-1]  # 移除 `:` 符号
            self.eat('LABEL_DEF')
            if self.current_token[0] == 'ASSIGN':
                self.eat('ASSIGN')  # 消耗 `;`，但不添加到 AST
        # 解析循环体
        loop_body = ASTNode("LoopBody", line=self.current_token[2], col=self.current_token[3])
        while self.current_token[0] != 'EOF' and self.current_token[0] != 'LABEL_DEF':
            loop_body.children.append(self.statement())
            if self.current_token[0] == 'ASSIGN':
                self.eat('ASSIGN')  # 消耗 `;`，但不添加到 AST
        # 寻找循环的结束条件
        if self.current_token[0] in ('BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'):
            condition = self.conditional_branch()
            if condition.children[1].value == start_label:
                end_label = self.current_token[1]
                self.eat('LABEL_DEF')
            else:
                raise SyntaxError(f"循环结束条件不匹配: 期望 {start_label}, 但得到 {condition.children[1].value} (行 {self.current_token[2]}, 列 {self.current_token[3]})")
        return ASTNode("Loop", None, [start_label, loop_body, end_label], line=self.current_token[2], col=self.current_token[3])
"""