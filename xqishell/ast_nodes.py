"""抽象语法树(AST)节点的唯一定义。

历史上 parser.py 与 evaluator.py 各有一份 ASTNode 定义且行为不一致
(evaluator 版会把 value=None 改写为 [])。现统一为:
value 缺省保持 None,children 缺省(含显式传 None)归一为空列表。
"""

from dataclasses import dataclass, field


@dataclass
class ASTNode:
    """抽象语法树的通用节点。"""

    type: str
    value: object = None
    children: list = field(default_factory=list)
    line: int | None = None
    col: int | None = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def __repr__(self):
        children_repr = [
            str(child) if isinstance(child, ASTNode) else repr(child)
            for child in self.children
        ]
        return f"ASTNode('{self.type}', value={self.value}, children=[{', '.join(children_repr)}])"
