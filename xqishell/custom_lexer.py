from pygments.lexer import RegexLexer
from prompt_toolkit.lexers import PygmentsLexer
from pygments.token import Text, Comment, Keyword, Name, String, Number, Punctuation

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
            (r'XQI\-BEGIN|XQI\-END|shot|error|ERR|U3|measure|CNOT|CMP|GPS|MOV|BX|BL|BEQ|BNE|BGT|BGE|BLT|BLE|ADD|SUB|MUL|DIV|LDR|STR|CLDR|CSTR|qreg|creg|reset|debug|debug\-p|rand|barrier|B', Keyword),
            (r'[a-zA-Z_][a-zA-Z0-9_]*\:', Keyword.Constant),
            (r'\[|\]|\{|\}|:|\(|\)|,|\.|;|\->', Punctuation),
            (r'LR|PC', Name),
            (r'"(\\\\|\\"|[^"])*"', String), #字符串类型，目前用不到
            (r"'(\\\\|\\'|[^'])*'", String), #字符串类型，目前用不到
            (r'.', Text)
        ],
    }
    def lex_document(self, document):
        lexer = PygmentsLexer(XQIASMLexer)  # 使用Prompt Toolkit的适配器
        return lexer.lex_document(document)

__all__ = ['XQIASMLexer']