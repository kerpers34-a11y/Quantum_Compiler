from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Keyword, Name, String, Number, Punctuation

from xqishell.instructions import opcode_alternation

# 关键字正则:指令清单一处定义(长度降序防前缀遮蔽,'-' 转义)
_KEYWORD_ALT = opcode_alternation().replace('-', r'\-')


class XQIASMLexer(RegexLexer):
    name = 'XQIASM'
    aliases = ['xqiasm']
    filenames = ['*.XQIASM']
    tokens = {
        'root': [
            (r'\s+', Text),
            (r'\;[^\n]*', Comment),
            (r'([-+]?\s*\d+(\.\d*)?\s*[-+])?\s*([-+]?\s*\d+(\.\d*)?\s*[ij])\s*([-+]\s*\d+(\.\d*)?)?', Number),
            (r'\b\d+(\.\d*)?\b', Number),
            (rf'XQI\-BEGIN|XQI\-END|{_KEYWORD_ALT}', Keyword),
            (r'[a-zA-Z_][a-zA-Z0-9_]*\:', Keyword.Constant),
            (r'\[|\]|\{|\}|:|\(|\)|,|\.|;|\->', Punctuation),
            (r'LR|PC', Name),
            (r'"(\\\\|\\"|[^"])*"', String), #字符串类型，目前用不到
            (r"'(\\\\|\\'|[^'])*'", String), #字符串类型，目前用不到
            (r'.', Text)
        ],
    }

__all__ = ['XQIASMLexer']
