"""XQIASM 错误类型体系(唯一来源)。

为兼容既有调用方,错误类多继承内置异常:
- XQISyntaxError 同时是 SyntaxError —— 老代码 `except SyntaxError` 仍能捕获;
- XQIExecutionError 同时是 ValueError —— 老代码 `except ValueError` 仍能捕获。

迁移约定:parser/xqi_lexer 抛 XQISyntaxError;evaluator 抛 XQIExecutionError
或其子类。错误消息保持既有中文原文。
"""


class XQIError(Exception):
    """XQIASM 所有自定义错误的基类。"""


class XQISyntaxError(XQIError, SyntaxError):
    """词法/语法分析阶段的错误。"""


class XQIExecutionError(XQIError, ValueError):
    """模拟执行阶段的错误。"""


class XQIInstructionError(XQIExecutionError):
    """与具体指令相关的执行错误,自动附加指令名前缀。"""

    def __init__(self, instr_name, msg):
        super().__init__(f"[{instr_name}] {msg}")
