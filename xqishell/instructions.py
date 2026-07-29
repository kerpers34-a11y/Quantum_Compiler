"""XQIASM 指令集的单一数据源。

词法分析(xqi_lexer)、终端高亮(custom_lexer)、Tab 补全(word_completer)、
行内建议(auto_suggest)统一从这里取指令清单,杜绝多处硬编码清单漂移。
"""

from dataclasses import dataclass

# 程序块标记(非指令,但参与补全与高亮)
BLOCK_MARKERS: tuple[str, str] = ("XQI-BEGIN", "XQI-END")

# 权威指令清单(34 条)
INSTRUCTIONS: tuple[str, ...] = (
    "shot", "error", "ERR", "U3", "measure", "CNOT", "CMP", "GPS", "MOV",
    "B", "BX", "BL", "BEQ", "BNE", "BGT", "BGE", "BLT", "BLE",
    "ADD", "SUB", "MUL", "DIV",
    "LDR", "STR", "CLDR", "CSTR",
    "qreg", "creg", "reset", "debug", "debug-p", "rand", "barrier",
)

# 条件/无条件分支(供 parser 语句分发等场景使用)
CONDITIONAL_BRANCHES = frozenset({"BEQ", "BNE", "BGT", "BGE", "BLT", "BLE"})
UNCONDITIONAL_BRANCHES = frozenset({"B", "BL"})


@dataclass(frozen=True)
class InstrMeta:
    """指令元数据。

    operands: 操作数签名示例(用于行内建议);
    category: 'pseudo' | 'gate' | 'measure' | 'memory' | 'arith' | 'control' | 'debug'
    """

    operands: str
    category: str


INSTR_META: dict[str, InstrMeta] = {
    # 伪指令
    "shot":    InstrMeta("shot 1", "pseudo"),
    "error":   InstrMeta("error(1)", "pseudo"),
    "ERR":     InstrMeta("ERR(1, R[n], R[m], 0.0, 0.0) q[a]", "pseudo"),
    "qreg":    InstrMeta("qreg q[n]", "pseudo"),
    "creg":    InstrMeta("creg c[n]", "pseudo"),
    "barrier": InstrMeta("barrier", "pseudo"),
    # 量子门
    "U3":      InstrMeta("U3(a,b,c) q[n]", "gate"),
    "CNOT":    InstrMeta("CNOT q[n],q[m]", "gate"),
    "GPS":     InstrMeta("GPS(delta) q[n]", "gate"),
    "reset":   InstrMeta("reset q[n]", "gate"),
    # 测量
    "measure": InstrMeta("measure q[n]->c[m]", "measure"),
    # 算术与传送
    "MOV":     InstrMeta("MOV R[n],R[m]", "arith"),
    "ADD":     InstrMeta("ADD R[d],R[n],R[m]", "arith"),
    "SUB":     InstrMeta("SUB R[d],R[n],R[m]", "arith"),
    "MUL":     InstrMeta("MUL R[d],R[n],R[m]", "arith"),
    "DIV":     InstrMeta("DIV R[d],R[n],R[m]", "arith"),
    "CMP":     InstrMeta("CMP R[n],R[m]", "arith"),
    "rand":    InstrMeta("rand R[d],R[s]", "arith"),
    # 内存访问
    "LDR":     InstrMeta("LDR R[n],M[m]", "memory"),
    "STR":     InstrMeta("STR R[n],M[m]", "memory"),
    "CLDR":    InstrMeta("CLDR c[n],M[re],M[im]", "memory"),
    "CSTR":    InstrMeta("CSTR c[n],M[re],M[im]", "memory"),
    # 控制流
    "B":       InstrMeta("B label", "control"),
    "BX":      InstrMeta("BX LR", "control"),
    "BL":      InstrMeta("BL label", "control"),
    "BEQ":     InstrMeta("BEQ label", "control"),
    "BNE":     InstrMeta("BNE label", "control"),
    "BGT":     InstrMeta("BGT label", "control"),
    "BGE":     InstrMeta("BGE label", "control"),
    "BLT":     InstrMeta("BLT label", "control"),
    "BLE":     InstrMeta("BLE label", "control"),
    # 调试
    "debug":   InstrMeta("debug", "debug"),
    "debug-p": InstrMeta("debug-p", "debug"),
}


def opcode_alternation() -> str:
    """生成正则交替串:按长度降序排列,避免前缀遮蔽。

    历史上 'debug' 排在 'debug-p' 之前,配合 (?=\\W) 断言导致
    'debug-p' 永远匹配不到(遮蔽即词法崩溃);'B' 与 'BX/BL/BEQ...' 同理。
    """
    return "|".join(sorted(INSTRUCTIONS, key=lambda s: (-len(s), s)))
