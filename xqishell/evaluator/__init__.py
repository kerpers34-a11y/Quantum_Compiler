"""双模式量子模拟器(statevector + density matrix)。

包化拆分(原单文件 evaluator.py):
- environment: QuantumEnvironment 量子态/寄存器/噪声应用
- noise: Kraus 算符构造(纯函数)
- executor: Evaluator 指令执行与程序求值(StateIO 混入)
- io: 状态/调试文件输出

对外兼容:`from xqishell.evaluator import Evaluator, QuantumEnvironment,
ASTNode, InstructionError` 等历史 import 路径全部保留。
"""

from xqishell.ast_nodes import ASTNode  # noqa: F401  # 兼容旧 import 路径
from xqishell.errors import XQIInstructionError

from .environment import QuantumEnvironment
from .executor import Evaluator

# 兼容旧名与计划中的新名
InstructionError = XQIInstructionError
XQIEvaluator = Evaluator

__all__ = [
    "ASTNode",
    "Evaluator",
    "XQIEvaluator",
    "QuantumEnvironment",
    "InstructionError",
]
