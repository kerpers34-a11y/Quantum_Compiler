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


class DebugInfoGenerator:
    def __init__(self, code):
        self.code_lines = code.split('\n')
        self.debug_message = []
        self.label_records = []
        self.xqi_begin_line = None
        self.xqi_end_line = None

        # 定位XQI区域
        self._locate_xqi_boundaries()
        # 生成调试信息
        self._build_debug_info()

    def _locate_xqi_boundaries(self):
        """定位XQI-BEGIN和XQI-END的物理行号"""
        for idx, line in enumerate(self.code_lines):
            stripped = line.strip()
            if stripped == "XQI-BEGIN":
                self.xqi_begin_line = idx
            elif stripped == "XQI-END":
                self.xqi_end_line = idx

        if self.xqi_begin_line is None:
            raise ValueError("Missing XQI-BEGIN declaration")
        if self.xqi_end_line is None:
            raise ValueError("Missing XQI-END declaration")
        if self.xqi_begin_line >= self.xqi_end_line:
            raise ValueError("XQI-BEGIN must come before XQI-END")

    def _process_single_line(self, raw_line):
        """处理单行内容并返回需要显示的信息"""
        stripped = raw_line.strip()

        # 处理空行
        if not stripped:
            return None

        # 分离注释和代码部分
        code_part, _, comment_part = stripped.partition(';')
        code_part = code_part.strip()

        # 处理标签定义
        label_match = re.match(r'^([a-zA-Z_]\w*):\s*$', code_part)
        if label_match:
            # 独立标签行处理
            self.label_records.append({
                'name': label_match.group(1),
                'sequence': len(self.label_records),
                'original_line': raw_line
            })
            return None

        # 处理包含标签的复合行
        label_colon = code_part.find(':')
        if label_colon != -1:
            label_name = code_part[:label_colon].strip()
            remaining_code = code_part[label_colon + 1:].strip()
            self.label_records.append({
                'name': label_name,
                'sequence': len(self.label_records),
                'original_line': raw_line
            })
            return remaining_code if remaining_code else None

        # 处理普通代码行
        display_code = code_part
        if comment_part:
            display_code += " ;"

        return display_code

    def _build_debug_info(self):
        """构建完整的调试信息结构"""
        self.debug_message = ["", "Operate Instructions:", "XQI-BEGIN"]
        sequence = 1

        # 处理XQI区域内的每一行
        for line in self.code_lines[self.xqi_begin_line + 1:self.xqi_end_line]:
            processed = self._process_single_line(line)

            if processed is not None:
                self.debug_message.append(f"     {sequence}. {processed}")
                sequence += 1
            elif line.strip() == '':  # 保留真实的空行
                self.debug_message.append("")

        # 添加XQI-END
        self.debug_message.append("XQI-END")

        # 添加标签表格
        if self.label_records:
            self.debug_message += [
                "",
                "Label Number   Sequence   Label Symbol",
                *[f"Label   {i:2d}:      {rec['sequence']:3d}        {rec['name']}"
                  for i, rec in enumerate(self.label_records)]
            ]

        # 转换为字符串
        self.debug_message = '\n'.join(self.debug_message)

# 解析器类
class XQILexer:
    def __init__(self, code):
        # 原有词法分析初始化
        token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_map)
        self.token_re = re.compile(token_regex)
        self.token_map = token_map
        self.code = code
        self.tokens = []

        # 独立生成调试信息
        self.debug_generator = DebugInfoGenerator(code)
        self.debug_message = self.debug_generator.debug_message

        # 原有词法分析流程
        self._tokenize()
        self._validate_xqi()

    def _validate_xqi(self):
        """验证XQI标记存在性"""
        has_begin = any(t[0] == 'XQI_BEGIN' for t in self.tokens)
        has_end = any(t[0] == 'XQI_END' for t in self.tokens)
        if not has_begin or not has_end:
            raise ValueError("Missing XQI boundary markers")

    def _tokenize(self):
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