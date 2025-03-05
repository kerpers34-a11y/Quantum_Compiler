import re
import config
class ASTNode:
    """ 抽象语法树的通用节点 """
    def __init__(self, type_, value=None, children=None, line=None, col=None):
        self.type = type_
        self.value = value
        self.children = children or []
        self.line = line
        self.col = col

    def __repr__(self):
        return f"{self.type}({self.value}, {self.children}, line={self.line}, col={self.col})"

    def convert_children_to_token(self, token_map):
        if not self.children:  # 如果是叶子节点
            if self.value is None:
                print(f"警告：节点值为 None，跳过匹配 (行 {self.line}, 列 {self.col})")
                return  # 直接返回，避免继续处理
            for token_name, pattern in token_map:
                if re.match(pattern, self.value):
                    self.children = [token_name]
                    break
        else:
            for child in self.children:
                if isinstance(child, ASTNode):
                    child.convert_children_to_token(token_map)


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
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

        # 检查 shot 是否在 XQI-BEGIN 的下一行
        if self.current_token[0] == 'OPCODE' and self.current_token[1] == 'shot':
            if self.current_token[2] != program_node.children[0].line + 1:
                raise SyntaxError(f"shot 必须在 XQI-BEGIN 的下一行 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            program_node.children.append(self.instruction())
        else:
            raise SyntaxError(f"shot 必须在 XQI-BEGIN 的下一行 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

        # 检查 error 是否在 shot 的下一行
        if self.current_token[0] == 'OPCODE' and self.current_token[1] == 'error':
            if self.current_token[2] != program_node.children[1].line + 1:
                raise SyntaxError(f"error 必须在 shot 的下一行 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            program_node.children.append(self.instruction())
        else:
            raise SyntaxError(f"error 必须在 shot 的下一行 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

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

        # 转换 children 为 token 名
        program_node.convert_children_to_token(self.token_map)

        return program_node

    def xqi_begin(self):
        """ 解析 XQI-BEGIN """
        node = ASTNode("XQI-BEGIN", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('XQI_BEGIN')
        # 忽略后面的分号
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')
        return node

    def xqi_end(self):
        """ 解析 XQI-END """
        node = ASTNode("XQI-END", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('XQI_END')
        # 忽略后面的分号
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')
        return node

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
        node = ASTNode("ProcessDefinition", self.current_token[1][:-1], line=self.current_token[2],
                       col=self.current_token[3])  # 移除 `:` 符号
        self.eat('LABEL_DEF')
        # 忽略后面的分号
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
            # 解析操作数
            operands_node = self.operand_list()
            self.validate_measure_operands(operands_node.children)
            # 检查 measure 是否包含 ->
            if not any(child.value == '->' for child in operands_node.children):
                raise SyntaxError(
                    f"语法错误: measure 指令缺乏 -> (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            node.children.append(opcode_node)
            node.children.append(operands_node)
        elif opcode_node.value == "shot":
            # 解析 shot 操作数
            operands_node = self.operand_list()
            if len(operands_node.children) != 1 or not operands_node.children[0].value.isdigit() or int(operands_node.children[0].value) <= 0:
                raise SyntaxError(
                    f"语法错误: shot 后面必须且仅能接一个正整数 (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            node.children.append(opcode_node)
            node.children.append(operands_node)
        elif opcode_node.value == "error":
            # 解析 error 操作数
            operands_node = self.operand_list()
            if not self.validate_error_operands(operands_node.children):
                 raise SyntaxError(
                    f"语法错误: error 操作数格式不正确 ({operands_node.children}行 {self.current_token[2]}, 列 {self.current_token[3]})")
            node.children.append(opcode_node)
            node.children.append(operands_node)
        elif opcode_node.value == "U3":
            operands_node = self.operand_list()
            self.validate_u3_operands(operands_node.children)
            node.children.extend([opcode_node, operands_node])
        elif opcode_node.value == "CNOT":
            operands_node = self.operand_list()
            self.validate_cnot_operands(operands_node.children)
            node.children.extend([opcode_node, operands_node])
        elif opcode_node.value in ["CLDR", "CSTR"]:
            operands_node = self.operand_list()
            self.validate_cldr_cstr_operands(opcode_node.value, operands_node.children)
            node.children.extend([opcode_node, operands_node])
        elif opcode_node.value in ["ADD", "SUB", "MUL", "DIV"]:  # 经典指令
            operands_node = self.operand_list()
            self.validate_classical_arithmetic_operands(operands_node.children)
            node.children.extend([opcode_node, operands_node])
        elif opcode_node.value == "MOV":
            operands_node = self.operand_list()
            self.validate_mov_operands(operands_node.children)
            node.children.extend([opcode_node, operands_node])
        else:
            # 其他指令的标准解析方式
            operands_node = self.operand_list()
            node.children.append(opcode_node)
            node.children.append(operands_node)
        # 确保 `;` 结束
        assign_node = self.eat('ASSIGN')
        node.children.append(ASTNode("ASSIGN", assign_node[1], line=assign_node[2], col=assign_node[3]))
        return node

    @staticmethod
    def handle_err_instruction(self, node, opcode_node):
        """ 处理ERR指令 """
        # 验证前序指令类型
        if self.last_valid_opcode not in ('U3', 'CNOT'):
            raise SyntaxError(f"ERR指令必须出现在U3或CNOT之后 (行 {opcode_node.line})")

        # 验证行号必须严格大于目标指令
        if opcode_node.line <= self.last_opcode_line:
            raise SyntaxError(
                f"ERR指令必须出现在目标指令之后 (当前行 {opcode_node.line}, 前序行 {self.last_opcode_line})")

        # 解析操作数
        operands_node = self.operand_list()

        # 参数结构验证
        params_node = next((c for c in operands_node.children if c.type == "Parameters"), None)
        if not params_node or len(params_node.children) != 5:
            raise SyntaxError(f"ERR指令需要5个参数 (行 {opcode_node.line})")

        # 参数值验证
        err_model = params_node.children[0].value
        err_prob = params_node.children[1].value
        if err_model not in {'1', '2', '3'}:
            raise SyntaxError(f"无效的错误模型类型 {err_model} (行 {opcode_node.line})")
        try:
            if not (0 <= float(err_prob) <= 1):
                raise ValueError
        except ValueError:
            raise SyntaxError(f"错误概率必须在0-1之间 (行 {opcode_node.line})")

        # 量子寄存器验证
        if not any(c.value.startswith('q[') for c in operands_node.children if c.type == "Operand"):
            raise SyntaxError(f"缺少量子寄存器参数 (行 {opcode_node.line})")

        # 构建节点
        node.children.extend([opcode_node, operands_node])
        self.eat('ASSIGN')  # 消耗分号

        # 更新最后有效指令状态
        self.last_valid_opcode = 'ERR'
        self.last_opcode_line = opcode_node.line

        return node

    @staticmethod
    def validate_error_operands(operands):
        """ 验证 error 操作数的格式 """
        # 展开 Parameters 节点
        expanded_operands = []
        for op in operands:
            if isinstance(op, ASTNode) and op.type == "Parameters":
                expanded_operands.extend(op.children)
            else:
                expanded_operands.append(op)

        def is_valid_num1(value):
            return value in ('0', '1')

        def is_valid_num2(value):
            return value.isdigit() and 0 <= int(value) <= 9

        def is_valid_num3(value):
            try:
                num = float(value)
                return 0.0 <= num <= 1.0
            except ValueError:
                return False

        # 检查展开后的操作数数量
        if len(expanded_operands) < 1 or len(expanded_operands) > 4:
            return False

        # 提取操作数的值
        values = []
        for operand in expanded_operands:
            if isinstance(operand, ASTNode) and operand.type == "Operand":
                values.append(operand.value)
            else:
                return False  # 无效的节点类型

        # 验证 num1
        num1 = values[0]
        if not is_valid_num1(num1):
            return False

        # 验证 num2（如果存在）
        if len(values) >= 2:
            num2 = values[1]
            if not is_valid_num2(num2):
                return False

        # 验证 num3（如果存在）
        if len(values) >= 3:
            num3 = values[2]
            if not is_valid_num3(num3):
                return False

        # 验证 num4（如果存在）
        if len(values) >= 4:
            num4 = values[3]
            if not is_valid_num3(num4):
                return False

        return True

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
        """ 解析操作数，支持 U3(参数, 参数, 参数) q[n]; 和 measure R[2] -> R[3]; """
        node = ASTNode("Operands", line=self.current_token[2], col=self.current_token[3])
        # 处理括号参数（如 U3(0.233, R[3], 0.14567)）
        if self.current_token[0] == 'LPAREN':
            self.eat('LPAREN')
            params = ASTNode("Parameters", line=self.current_token[2], col=self.current_token[3])
            while self.current_token[0] not in ('RPAREN', 'EOF'):
                params.children.append(self.operand())
                if self.current_token[0] == 'COMMA':
                    self.eat('COMMA')
            self.eat('RPAREN')  # 消耗 )
            node.children.append(params)
        # 处理普通操作数（寄存器、立即数、标签等）
        operands = []
        while self.current_token[0] in ('REGISTER', 'REGISTER_M', 'REGISTER_C', 'REGISTER_Q',
                                        'REGISTER_LR', 'REGISTER_PC', 'IMMEDIATE', 'NUMBER', 'COMPLEX','LABEL'):
            operands.append(self.operand())
            if self.current_token[0] == 'COMMA':  # 处理 , 逗号分隔的操作数
                self.eat('COMMA')
        # 处理 -> 语法（将 `->` 视为 `,` 逗号，使 measure 解析方式与 ADD 统一）
        if self.current_token[0] == 'ARROW':  # 检测 ->
            self.eat('ARROW')
            operands.append(ASTNode("ARROW", "->", line=self.current_token[2], col=self.current_token[3]))  # 添加 -> 作为节点
            operands.append(self.operand())  # 解析箭头后面的目标寄存器
        node.children.extend(operands)
        return node

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
        """ 解析无条件跳转和过程调用 (B label; or BL label;) """
        opcode = ASTNode("Opcode", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('OPCODE')
        label = ASTNode("Label", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('LABEL')
        self.eat('ASSIGN')
        return ASTNode("UnconditionalBranch", None, [opcode, label], line=self.current_token[2], col=self.current_token[3])

    def conditional_branch(self):
        """ 解析条件分支 (BEQ|BNE|BGT|BGE|BLT|BLE label;) """
        opcode = ASTNode("Opcode", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('OPCODE')
        label = ASTNode("Label", self.current_token[1], line=self.current_token[2], col=self.current_token[3])
        self.eat('LABEL')
        self.eat('ASSIGN')
        return ASTNode("ConditionalBranch", None, [opcode, label], line=self.current_token[2], col=self.current_token[3])

    @staticmethod
    def validate_mov_operands(operands):
        for op in operands:
            if isinstance(op, ASTNode) and op.type == "Operand":
                if op.value.startswith(('q[', 'c[', 'M[')):
                    raise SyntaxError(f"MOV指令禁止使用 {op.value} 寄存器 (行 {op.line})")

    @staticmethod
    def validate_measure_operands(operands):
        arrow_index = None
        for i, op in enumerate(operands):
            if isinstance(op, ASTNode) and op.value == '->':
                arrow_index = i
                break
        if arrow_index is None or arrow_index != 1:
            raise SyntaxError("measure指令必须包含 -> 分隔符")
        q_register = operands[0]
        c_register = operands[2]
        if not q_register.value.startswith('q['):
            raise SyntaxError(f"measure源操作数必须是量子寄存器 (行 {q_register.line})")
        if not c_register.value.startswith('c['):
            raise SyntaxError(f"measure目标操作数必须是经典寄存器 (行 {c_register.line})")

    def validate_u3_operands(self, operands):
        # 提取参数节点
        params_node = next((c for c in operands if isinstance(c, ASTNode) and c.type == "Parameters"), None)

        # 验证参数数量
        if not params_node or len(params_node.children) != 3:
            raise SyntaxError(f"U3指令需要3个参数 (行 {self.current_token[2]})")

        # 验证参数类型（可以是数字、立即数或寄存器）
        for param in params_node.children:
            if param.type not in ("Operand", "IMMEDIATE", "NUMBER"):
                raise SyntaxError(f"U3参数 {param.value} 类型无效 (行 {param.line})")

        # 验证量子寄存器操作数
        q_operands = [
            op for op in operands
            if isinstance(op, ASTNode) and op.type == "Operand" and op.value.startswith('q[')
        ]
        if len(q_operands) != 1:
            raise SyntaxError(f"U3需要且只能指定一个量子寄存器 (行 {self.current_token[2]})")

        # 确保没有多余操作数
        if len(operands) != 2:  # 1个Parameters节点 + 1个Operand节点
            raise SyntaxError(f"U3语法格式错误，应为 U3(theta,phi,lambda) q[n]; (行 {self.current_token[2]})")

    @staticmethod
    def validate_cnot_operands(operands):
        if len(operands) < 2:
            raise SyntaxError("CNOT需要两个量子寄存器")
        for op in operands[:2]:
            if not op.value.startswith('q['):
                raise SyntaxError(f"CNOT操作数 {op.value} 不是量子寄存器 (行 {op.line})")

    def validate_gps_operands(self, operands):
        """ 验证GPS指令格式：GPS(delta) q[m]; """
        # 提取参数节点和操作数
        params_node = next((c for c in operands if c.type == "Parameters"), None)
        operands_list = [op for op in operands if op.type == "Operand"]

        # 验证参数部分
        if not params_node or len(params_node.children) != 1:
            raise SyntaxError(f"GPS指令必须包含一个参数delta (行 {self.current_token[2]})")
        delta_param = params_node.children[0]
        if delta_param.value.startswith('q['):
            raise SyntaxError(f"GPS参数delta不能是量子寄存器 (行 {delta_param.line})")

        # 验证操作数部分
        if len(operands_list) != 1 or not operands_list[0].value.startswith('q['):
            raise SyntaxError(f"GPS第二个操作数必须是量子寄存器 (行 {self.current_token[2]})")

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