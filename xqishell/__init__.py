from pathlib import Path
from pygments.lexers import load_lexer_from_file

# 导入本包的其他模块
from .style import style_prompt, message_prompt, style_html
from .word_completer import opcode_completer
from .auto_suggest import CustomAutoSuggest

# 正确获取包内文件路径
_here = Path(__file__).parent
_custom_lexer_path = _here / "custom_lexer.py"

xqiasm_lexer = load_lexer_from_file(
    filename=str(_custom_lexer_path),
    lexername="XQIASMLexer",
    fullname="XQIASM Lexer"
)
