"""词法单元(Token)的唯一定义。

Token 是 NamedTuple(tuple 子类)。历史上下游以 `token[0]..token[3]`
索引访问,该方式保持兼容;新代码应优先使用 `.type/.value/.line/.col`
属性访问以提升可读性。
"""

from typing import NamedTuple


class Token(NamedTuple):
    type: str
    value: str
    line: int
    col: int
