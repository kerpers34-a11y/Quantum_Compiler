"""对外兼容性测试:历史 import 路径、config 旧名 alias、Token 索引访问。

重构的兼容承诺(见计划 §1):
- `from xqishell.parser import Parser, ASTNode` 可用;
- `from xqishell.evaluator import Evaluator, QuantumEnvironment, ASTNode, InstructionError` 可用;
- `xqishell.main:main` 入口不变,`main_progress` 旧名可用;
- config 旧常量名全部保留为 alias 且取值不变;
- Token 同时支持索引访问与属性访问。
"""

from xqishell import config
from xqishell.tokens import Token


def test_parser_module_reexports():
    from xqishell.parser import ASTNode, Parser  # noqa: F401

    node = ASTNode("Instruction", children=None)  # 显式 None 归一为空列表
    assert node.children == []
    assert node.value is None  # 统一为 None,不再是 []


def test_evaluator_module_reexports():
    from xqishell.evaluator import ASTNode, Evaluator, InstructionError, QuantumEnvironment  # noqa: F401

    assert issubclass(InstructionError, ValueError)  # 旧兼容:仍可按 ValueError 捕获


def test_error_hierarchy_builtin_compat():
    from xqishell.errors import XQIExecutionError, XQISyntaxError

    assert issubclass(XQISyntaxError, SyntaxError)
    assert issubclass(XQIExecutionError, ValueError)


def test_main_entry_and_legacy_alias():
    from xqishell.main import main, main_progress, run_with_progress  # noqa: F401

    assert main_progress is run_with_progress


def test_config_legacy_aliases():
    assert config.MAX_Classical_Register == config.MAX_CLASSICAL_REGISTER == 8
    assert config.MAX_shot_TIMES == config.MAX_SHOT_TIMES == 8192
    assert config.MAX_Register == config.MAX_REGISTER == 16
    assert config.MAX_Memory == config.MAX_MEMORY == 32
    assert config.true is True and config.false is False
    assert config.inf == 0x7FFFFFFF  # C 版 INT_MAX 语义,不是数学无穷大
    assert config.pi == 3.14159265  # C 版截断精度,不是 math.pi
    assert config.filename_debug == "XQI-QC-list.txt"
    assert config.default_Q_error_Code == config.DEFAULT_Q_ERROR_CODE == 1
    assert config.Length_string_XQI_BEGIN == config.LENGTH_STRING_XQI_BEGIN == 9


def test_token_index_and_attribute_access():
    tok = Token("OPCODE", "MOV", 3, 5)
    assert tok[0] == "OPCODE" and tok[1] == "MOV" and tok[2] == 3 and tok[3] == 5
    assert tok.type == "OPCODE" and tok.value == "MOV" and tok.line == 3 and tok.col == 5
    assert tuple(tok) == ("OPCODE", "MOV", 3, 5)


def test_ensure_xqi_tags_behavior():
    from xqishell.textprocessing_funs import ensure_xqi_tags

    # 缺失标记时补齐,并以换行结尾
    assert ensure_xqi_tags("MOV R[0],1;") == "XQI-BEGIN\nMOV R[0],1;\nXQI-END\n"
    # 已有标记时不重复添加;注意:输入以换行结尾时会再追加一个换行
    # (行为锁定,记录现状而非断言理想行为)
    already = "XQI-BEGIN\nMOV R[0],1;\nXQI-END\n"
    assert ensure_xqi_tags(already) == already + "\n"
