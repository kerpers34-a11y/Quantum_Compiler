import re
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
            self.eat('ASSIGN')  # 消耗 `;`，但不添加到 AST
        else:
            raise SyntaxError(f"程序必须以 XQI-BEGIN 开始 (行 {self.current_token[2]}, 列 {self.current_token[3]})")

        while self.current_token[0] != 'EOF' and self.current_token[0] != 'XQI_END':
            program_node.children.append(self.statement())

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
        if opcode_node.value == "measure":
            # 解析操作数
            operands_node = self.operand_list()
            # 检查 measure 是否包含 ->
            if not any(child.value == '->' for child in operands_node.children):
                raise SyntaxError(
                    f"语法错误: measure 指令缺乏 -> (行 {self.current_token[2]}, 列 {self.current_token[3]})")
            node.children.append(opcode_node)
            node.children.append(operands_node)
        else:
            # 其他指令的标准解析方式
            operands_node = self.operand_list()
            node.children.append(opcode_node)
            node.children.append(operands_node)
        # 确保 `;` 结束
        assign_node = self.eat('ASSIGN')
        node.children.append(ASTNode("ASSIGN", assign_node[1], line=assign_node[2], col=assign_node[3]))
        return node

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
                                        'REGISTER_LR', 'REGISTER_PC', 'IMMEDIATE', 'LABEL'):
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
            node = ASTNode("Operand", f"R[{token[1][2:-1]}]", line=token[2], col=token[3])  # 提取 R[n] 格式
        elif token[0] == 'REGISTER_M':
            node = ASTNode("Operand", f"M[{token[1][2:-1]}]", line=token[2], col=token[3])  # 提取 M[n] 格式
        elif token[0] == 'REGISTER_C':
            node = ASTNode("Operand", f"c[{token[1][2:-1]}]", line=token[2], col=token[3])  # 提取 c[n] 格式
        elif token[0] == 'REGISTER_Q':
            node = ASTNode("Operand", f"q[{token[1][2:-1]}]", line=token[2], col=token[3])  # 提取 q[n] 格式
        elif token[0] == 'REGISTER_LR':
            node = ASTNode("Operand", "LR", line=token[2], col=token[3])
        elif token[0] == 'REGISTER_PC':
            node = ASTNode("Operand", "PC", line=token[2], col=token[3])
        elif token[0] == 'IMMEDIATE':
            node = ASTNode("Operand", token[1], line=token[2], col=token[3])  # 立即数
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

    def loop(self):
        """ 解析循环结构 """
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
