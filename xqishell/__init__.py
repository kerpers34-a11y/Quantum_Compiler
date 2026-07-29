# 导入本包的其他模块
from .auto_suggest import CustomAutoSuggest
from .custom_lexer import XQIASMLexer
from .style import ASCII_ART_LOGO, message_prompt, style_html, style_prompt
from .word_completer import opcode_completer

# 终端语法高亮使用的 Pygments lexer 类(与旧 load_lexer_from_file 动态加载等价)
xqiasm_lexer = XQIASMLexer

__all__ = [
    "ASCII_ART_LOGO",
    "CustomAutoSuggest",
    "XQIASMLexer",
    "message_prompt",
    "opcode_completer",
    "style_html",
    "style_prompt",
    "xqiasm_lexer",
]
