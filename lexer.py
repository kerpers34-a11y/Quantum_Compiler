import re

token_map = [
    ('COMMENT', r'\;[^\n]*'),  # 注释：以 ';' 开始直到换行
    ('XQI_BEGIN', r'XQI\-BEGIN'),
    ('XQI_END', r'XQI\-END'),
    ('OPCODE', r'(?<!\w)(shot|error|ERR|U3|measure|CNOT|CMP|GPS|MOV|B|BX|BL|BEQ|BNE|BGT|BGE|BLT|BLE|ADD|SUB|MUL|DIV|LDR|STR|CLDR|CSTR|qreg|creg|reset|debug|debug\-p|rand|barrier)(?=\W)'),
    ('REGISTER', r'R\[\d+\]'),  # R[n]寄存器
    ('REGISTER_M', r'M\[\d+\]'),  # M[n]寄存器
    ('REGISTER_C', r'c\[\d+\]'),  # c[n]寄存器
    ('REGISTER_Q', r'q\[\d+\]'),  # q[n]寄存器
    ('REGISTER_LR', r'LR'),  # LR寄存器
    ('REGISTER_PC', r'PC'),  # PC寄存器
    ('ARROW', r'\->'),
    ('COMPLEX', r'([-+]?\s*\d+(\.\d*)?\s*[-+])?\s*([-+]?\s*\d+(\.\d*)?\s*[ij])\s*([-+]\s*\d+(\.\d*)?)?'),
    ('NUMBER', r'\b\d+(\.\d*)?\b'),
    ('IMMEDIATE', r'#\d+'),
    ('LABEL_DEF', r'[a-zA-Z_][a-zA-Z0-9_]*\:'),  # 过程定义 (label:)
    ('LABEL', r'[a-zA-Z_][a-zA-Z0-9_]*'),  # 标签引用
    ('LPAREN', r'\('), ('RPAREN', r'\)'),
    ('ASSIGN', r'\;'),
    ('COMMA', r'\,'),
    ('COLON', r'\:'),  # 仅在标签定义中使用
    ('SKIP', r'[ \t]+'),
    ('NEWLINE', r'\n'),
    ('MISMATCH', r'.'),
]

# 解析器类
class Lexer:
    def __init__(self, code):
        token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_map)
        self.token_re = re.compile(token_regex)
        self.code = code
        self.tokens = []
        self.token_map = token_map
        self.tokenize()
        # 检查是否存在 XQI-BEGIN 和 XQI-END
        if not any(token[0] == 'XQI_BEGIN' for token in self.tokens):
            raise ValueError("错误：缺少 'XQI-BEGIN'")
        if not any(token[0] == 'XQI_END' for token in self.tokens):
            raise ValueError("错误：缺少 'XQI-END'")

    def tokenize(self):
        line = 1
        col = 1
        for match in self.token_re.finditer(self.code):
            if match is None:
                raise ValueError(f"正则匹配失败 (行 {line}, 列 {col})")

            kind = match.lastgroup
            value = match.group()
            if kind in ('SKIP', 'NEWLINE'):
                if kind == 'NEWLINE':
                    line += 1
                    col = 1
                else:
                    col += len(value)
                continue  # 忽略空白字符
            elif kind == 'COMMENT':
                # 保留分号
                self.tokens.append(('ASSIGN', ';', line, col))
                col += 1  # 分号的长度
                col += len(value) - 1  # 注释的其余部分
                continue  # 忽略注释的其余部分
            elif kind == 'MISMATCH':
                raise ValueError(f"非法字符: {value} (行 {line}, 列 {col})")
            self.tokens.append((kind, value, line, col))
            col += len(value)
        # 确保 `;` 始终作为单独的 Token
        if self.tokens and self.tokens[-1][0] != 'ASSIGN':
            self.tokens.append(('ASSIGN', ';', line, col))

    def next_token(self):
        if self.tokens:
            token = self.tokens.pop(0)
            return token
        else:
            return 'EOF', 'EOF', 0, 0