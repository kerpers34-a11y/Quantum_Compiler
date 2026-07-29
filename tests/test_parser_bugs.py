"""已实锤 bug 的回归测试(parser 侧)。

B1: validate_gps_operands 的 delta/qubit 两个检查写反(parser.py:524-542),
    导致合法 `GPS(delta) q[n];` 必报语法错误,且 `len(ASTNode)` 会 TypeError;
B8: handle_err_instruction 是 @staticmethod 却声明 self(parser.py:247-248),
    调用处显式传 self,能跑但极度误导;
B9(测试编写期新发现): program() 的 shot/error 前置扫描循环只推进 NEWLINE/OPCODE
    两类 token,遇到 LABEL_DEF 等其他 token 且 shot/error 未齐时死循环
    (含标签但缺 error 声明的程序会把解释器挂死)。

修复前:test_b1_gps_valid_syntax_accepted / test_b1_gps_rejects_non_qreg_second_operand /
test_b8_handle_err_is_plain_method / test_b9 应为红;test_b1_gps_delta_rejects_qreg
为行为守护(修复前后都应抛 delta 错,只是当前实现抛错路径是错的)。

注意:XQIASM 程序必须声明 shot 与 error(parser 强制),error(0) 表示关闭噪声模型。
"""

import inspect
import subprocess
import sys

import pytest

from xqishell.parser import Parser
from xqishell.xqi_lexer import XQILexer


def parse(src: str):
    return Parser(XQILexer(src)).program()


def wrap(body: str) -> str:
    return f"XQI-BEGIN\nshot 1;\nerror(0);\nqreg q[2];\ncreg c[2];\n{body}\nXQI-END\n"


@pytest.mark.bug
def test_b1_gps_valid_syntax_accepted():
    """合法 GPS(0.5) q[0]; 不应抛任何异常(当前误报 delta 不能是量子寄存器)。"""
    parse(wrap("GPS(0.5) q[0];"))


@pytest.mark.bug
def test_b1_gps_delta_rejects_qreg():
    """delta 位置放量子寄存器必须报 delta 错(行为守护,修复前后均成立)。"""
    with pytest.raises(SyntaxError, match="delta"):
        parse(wrap("GPS(q[0]) q[1];"))


@pytest.mark.bug
def test_b1_gps_rejects_non_qreg_second_operand():
    """第二操作数非量子寄存器必须报 SyntaxError(当前 len(ASTNode) 抛 TypeError)。"""
    with pytest.raises(SyntaxError, match="第二个操作数"):
        parse(wrap("GPS(0.5) 5;"))


@pytest.mark.bug
def test_b8_handle_err_is_plain_method():
    """handle_err_instruction 不应是 staticmethod(当前 @staticmethod 带 self 怪胎)。"""
    assert not isinstance(inspect.getattr_static(Parser, "handle_err_instruction"), staticmethod)


@pytest.mark.bug
def test_b9_parser_no_hang_on_missing_error_with_labels():
    """含标签但缺 error 声明的程序必须快速报 SyntaxError,而不是死循环挂死。

    当前实现会在 program() 前置扫描中死循环,因此在子进程中运行并限时:
    修复前子进程超时(红),修复后迅速以 SyntaxError 退出(绿)。
    """
    program = (
        "import sys; sys.path.insert(0, r'%s');"
        "from xqishell.xqi_lexer import XQILexer;"
        "from xqishell.parser import Parser;"
        "src = 'XQI-BEGIN\\nshot 1;\\nqreg q[1];\\ncreg c[1];\\nBEQ equal;\\nequal: MOV PC,0;\\nXQI-END\\n';"
        "Parser(XQILexer(src)).program()"
    ) % str(__import__("pathlib").Path(__file__).resolve().parents[1])
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            timeout=15,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("parser 在缺少 error 声明且含标签的程序上死循环(挂死超过 15s)")
    assert result.returncode != 0
    assert "error" in result.stderr
