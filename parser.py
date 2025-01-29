class ASTNode:
    """ 抽象语法树的通用节点 """
    def __init__(self, type_, value=None, children=None):
        self.type = type_
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"{self.type}({self.value}, {self.children})"

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.next_token()

    def eat(self, token_type):
        """ 消耗当前 token，并获取下一个 token """
        if self.current_token[0] == token_type:
            self.current_token = self.lexer.next_token()
        else:
            raise SyntaxError(f"语法错误: 期望 {token_type}, 但得到 {self.current_token}")

    def program(self):
        """ 解析完整的程序，支持顺序执行、分支、循环 """
        program_node = ASTNode("Program")

        # 确保 XQI-BEGIN 只出现在第一行
        if self.current_token[0] == 'XQI_BEGIN':
            program_node.children.append(self.xqi_begin())
            self.eat('ASSIGN')  # 消耗 `;`
        else:
            raise SyntaxError("程序必须以 XQI-BEGIN 开始")

        while self.current_token[0] != 'EOF' and self.current_token[0] != 'XQI_END':
            program_node.children.append(self.statement())

        # 确保 XQI-END 只出现在最后一行
        if self.current_token[0] == 'XQI_END':
            program_node.children.append(self.xqi_end())
            self.eat('ASSIGN')  # 消耗 `;`
        else:
            raise SyntaxError("程序必须以 XQI-END 结束")

        return program_node

    def xqi_begin(self):
        """ 解析 XQI-BEGIN """
        node = ASTNode("XQI-BEGIN", self.current_token[1])
        self.eat('XQI_BEGIN')
        return node

    def xqi_end(self):
        """ 解析 XQI-END """
        node = ASTNode("XQI-END", self.current_token[1])
        self.eat('XQI_END')
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
            raise SyntaxError("孤立的 '->' 语法错误，可能缺少 `measure` 指令")
        else:
            raise SyntaxError(f"无效的语句: {self.current_token}")

    def process_definition(self):
        """ 解析过程定义 (label:) """
        node = ASTNode("ProcessDefinition", self.current_token[1][:-1])  # 移除 `:` 符号
        self.eat('LABEL_DEF')
        return node

    def instruction(self):
        """ 解析指令，确保 measure 作为独立的 Instruction """
        node = ASTNode("Instruction")
        opcode_node = self.opcode()

        if opcode_node.value == "measure":
            # 解析操作数
            operands_node = self.operand_list()

            # 检查 measure 是否包含 ->
            if not any(child.value == '->' for child in operands_node.children):
                raise SyntaxError("语法错误: measure 指令缺乏 ->")

            node.children.append(opcode_node)
            node.children.append(operands_node)
        else:
            # 其他指令的标准解析方式
            operands_node = self.operand_list()
            node.children.append(opcode_node)
            node.children.append(operands_node)

        return node

    def opcode(self):
        """ 解析操作码 """
        token = self.current_token
        node = ASTNode("Opcode", token[1])
        self.eat('OPCODE')
        return node

    def operand_list(self):
        """ 解析操作数，支持 U3(参数, 参数, 参数) q[n]; 和 measure R[2] -> R[3]; """
        node = ASTNode("Operands")

        # 处理括号参数（如 U3(0.233, R[3], 0.14567)）
        if self.current_token[0] == 'LPAREN':
            self.eat('LPAREN')
            params = ASTNode("Parameters")
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
            operands.append(ASTNode("ARROW", "->"))  # 添加 -> 作为节点
            operands.append(self.operand())  # 解析箭头后面的目标寄存器
            node.children.extend(operands)
        else:
            node.children.extend(operands)

        # 确保 `;` 结束
        if self.current_token[0] == 'ASSIGN':
            self.eat('ASSIGN')

        return node

    def operand(self):
        """ 解析单个操作数 """
        token = self.current_token
        if token[0] == 'REGISTER':
            node = ASTNode("Operand", f"R[{token[1][2:-1]}]")  # 提取 R[n] 格式
        elif token[0] == 'REGISTER_M':
            node = ASTNode("Operand", f"M[{token[1][2:-1]}]")  # 提取 M[n] 格式
        elif token[0] == 'REGISTER_C':
            node = ASTNode("Operand", f"c[{token[1][2:-1]}]")  # 提取 c[n] 格式
        elif token[0] == 'REGISTER_Q':
            node = ASTNode("Operand", f"q[{token[1][2:-1]}]")  # 提取 q[n] 格式
        elif token[0] == 'REGISTER_LR':
            node = ASTNode("Operand", "LR")
        elif token[0] == 'REGISTER_PC':
            node = ASTNode("Operand", "PC")
        else:
            node = ASTNode("Operand", token[1])  # 其他类型的操作数

        self.eat(token[0])
        return node

    def unconditional_branch(self):
        """ 解析无条件跳转和过程调用 (B label; or BL label;) """
        opcode = ASTNode("Opcode", self.current_token[1])
        self.eat('OPCODE')
        label = ASTNode("Label", self.current_token[1])
        self.eat('LABEL')
        self.eat('ASSIGN')
        return ASTNode("UnconditionalBranch", None, [opcode, label])

    def conditional_branch(self):
        """ 解析条件分支 (BEQ|BNE|BGT|BGE|BLT|BLE label;) """
        opcode = ASTNode("Opcode", self.current_token[1])
        self.eat('OPCODE')
        label = ASTNode("Label", self.current_token[1])
        self.eat('LABEL')
        self.eat('ASSIGN')
        return ASTNode("ConditionalBranch", None, [opcode, label])

    def loop(self):
        """ 解析循环结构 """
        start_label = None
        end_label = None

        # 寻找循环的开始标签
        if self.current_token[0] == 'LABEL_DEF':
            start_label = self.current_token[1][:-1]  # 移除 `:` 符号
            self.eat('LABEL_DEF')
            self.eat('NEWLINE')

        # 解析循环体
        loop_body = ASTNode("LoopBody")
        while self.current_token[0] != 'EOF' and self.current_token[0] != 'LABEL_DEF':
            loop_body.children.append(self.statement())
            self.eat('NEWLINE')

        # 寻找循环的结束条件
        if self.current_token[0] in ('BEQ', 'BNE', 'BGT', 'BGE', 'BLT', 'BLE'):
            condition = self.conditional_branch()
            if condition.children[1].value == start_label:
                end_label = self.current_token[1]
                self.eat('LABEL_DEF')
            else:
                raise SyntaxError(f"循环结束条件不匹配: 期望 {start_label}, 但得到 {condition.children[1].value}")

        return ASTNode("Loop", None, [start_label, loop_body, end_label])