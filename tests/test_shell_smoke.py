"""Shell(REPL)侧冒烟与结构性回归测试。

B5: main.py 存在两个 `elif user_input.startswith("cat ")` 分支(L324/L364),
    第二个永不执行且实现不一致。
Phase 4 命令分发表落地后,cat 应成为 COMMANDS 中的唯一条目,
本测试同时兼容两种实现形态。
"""

import os
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT
from xqishell.commands import COMMANDS, ShellContext, dispatch

XQISHELL_DIR = PROJECT_ROOT / "xqishell"


@pytest.mark.bug
def test_b5_single_cat_command_handler():
    """cat 命令处理必须全仓唯一(当前 main.py 两处 startswith 分支)。"""
    startswith_hits = 0
    dispatch_hits = 0
    for p in XQISHELL_DIR.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        # 兼容单双引号两种写法
        startswith_hits += text.count("startswith('cat ')") + text.count('startswith("cat ")')
        # Phase 4 命令分发表形态:"cat": cmd_xxx 或 'cat': cmd_xxx
        dispatch_hits += text.count('"cat":') + text.count("'cat':")
    total = startswith_hits + dispatch_hits
    assert total == 1, f"cat 命令处理出现 {total} 处,必须唯一"


def make_ctx(calls=None):
    """构造测试用上下文:run_program 记录调用,multi_line_input 返回 None。"""
    if calls is None:
        calls = []
    return ShellContext(run_program=lambda s: calls.append(s), multi_line_input=lambda **kw: None), calls


@pytest.fixture(autouse=True)
def _mute_command_print(monkeypatch):
    """打桩命令层的样式化输出:pytest 无控制台,print_formatted_text 会抛
    NoConsoleScreenBufferError;冒烟测试关注行为而非终端渲染。"""
    monkeypatch.setattr("xqishell.commands._print", lambda *a, **k: None)


def test_dispatch_cat_reads_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello-xqi", encoding="utf-8")
    ctx, _ = make_ctx()
    assert dispatch(ctx, "cat a.txt") is True


def test_dispatch_mkdir_and_rm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx, _ = make_ctx()
    assert dispatch(ctx, "mkdir sub") is True
    assert (tmp_path / "sub").is_dir()
    assert dispatch(ctx, "rm sub") is True
    assert not (tmp_path / "sub").exists()


def test_dispatch_mv_and_cp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    ctx, _ = make_ctx()
    assert dispatch(ctx, "cp a.txt b.txt") is True
    assert (tmp_path / "b.txt").read_text() == "x"
    assert dispatch(ctx, "mv b.txt c.txt") is True
    assert (tmp_path / "c.txt").exists() and not (tmp_path / "b.txt").exists()


def test_dispatch_execute_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "P.XQIASM").write_text("XQI-BEGIN\nXQI-END\n", encoding="utf-8")
    ctx, calls = make_ctx()
    assert dispatch(ctx, "./P.XQIASM") is True
    assert calls == ["XQI-BEGIN\nXQI-END\n"]


def test_dispatch_execute_file_missing(tmp_path, monkeypatch):
    """./不存在.XQIASM 不抛异常,不调用 run_program。"""
    monkeypatch.chdir(tmp_path)
    ctx, calls = make_ctx()
    assert dispatch(ctx, "./missing.XQIASM") is True
    assert calls == []


def test_dispatch_unknown_and_argless_are_silent():
    ctx, _ = make_ctx()
    assert dispatch(ctx, "foobar") is False
    assert dispatch(ctx, "cat") is False  # 缺参数:对齐原 startswith('cat ') 不匹配行为
    assert "cat" in COMMANDS and len(COMMANDS) >= 9
