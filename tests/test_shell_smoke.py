"""Shell(REPL)侧冒烟与结构性回归测试。

B5: main.py 存在两个 `elif user_input.startswith("cat ")` 分支(L324/L364),
    第二个永不执行且实现不一致。
Phase 4 命令分发表落地后,cat 应成为 COMMANDS 中的唯一条目,
本测试同时兼容两种实现形态。
"""

from pathlib import Path

import pytest

from conftest import PROJECT_ROOT

XQISHELL_DIR = PROJECT_ROOT / "xqishell"


@pytest.mark.bug
def test_b5_single_cat_command_handler():
    """cat 命令处理必须全仓唯一(当前 main.py 两处 startswith 分支)。"""
    startswith_hits = 0
    dispatch_hits = 0
    for p in XQISHELL_DIR.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        startswith_hits += text.count('startswith("cat ")')
        # Phase 4 命令分发表形态:"cat": cmd_xxx 或 'cat': cmd_xxx
        dispatch_hits += text.count('"cat":') + text.count("'cat':")
    total = startswith_hits + dispatch_hits
    assert total == 1, f"cat 命令处理出现 {total} 处,必须唯一"
