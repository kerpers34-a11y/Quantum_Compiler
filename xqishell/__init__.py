from prompt_toolkit.lexers import PygmentsLexer

# 导入本包的其他模块
from .auto_suggest import CustomAutoSuggest
from .custom_lexer import XQIASMLexer
from .style import ASCII_ART_LOGO, message_prompt, style_html, style_prompt
from .word_completer import opcode_completer

# 终端语法高亮:prompt_toolkit 的 lexer= 需要其实现 lex_document() 的 Lexer 实例,
# 故用 PygmentsLexer 包装 Pygments lexer 类(与旧 load_lexer_from_file 动态加载等价)
xqiasm_lexer = PygmentsLexer(XQIASMLexer)

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
