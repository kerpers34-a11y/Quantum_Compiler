from pygments.lexers import _lexer_cache, load_lexer_from_file
from .style import style_prompt, message_prompt, style_html
from .custom_lexer import XQIASMLexer
from .word_completer import opcode_completer
from .auto_suggest import CustomAutoSuggest

xqiasm_lexer  = load_lexer_from_file(
    filename=r"xqi_shell/custom_lexer.py",  # 文件路径
    lexername="XQIASMLexer",                # Lexer类名
    fullname="XQIASM Lexer"                 # 显示名称
)
