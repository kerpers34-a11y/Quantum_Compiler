# 导入本包的其他模块
from .custom_lexer import XQIASMLexer
from .style import ASCII_ART_LOGO, style_prompt, message_prompt, style_html
from .word_completer import opcode_completer
from .auto_suggest import CustomAutoSuggest

# 终端语法高亮使用的 Pygments lexer 类(与旧 load_lexer_from_file 动态加载等价)
xqiasm_lexer = XQIASMLexer
