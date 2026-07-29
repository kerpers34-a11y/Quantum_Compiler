"""XQI Shell 的 REPL 命令处理与分发表。

原 main() 中约 15 个 startswith/elif 分支的上帝函数,拆为:
- 每个命令一个独立处理函数;
- COMMANDS 分发表:首词精确查表,消除 startswith 链与顺序敏感问题;
- dispatch() 供 main 的 prompt 循环调用。

参数解析沿用朴素的空白切分(与历史行为一致;不支持引号,
Windows 反斜杠路径不受影响)。
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit import HTML, print_formatted_text

from xqishell import style_html


def _print(html_text):
    print_formatted_text(HTML(html_text), style=style_html)


@dataclass
class ShellContext:
    """REPL 运行上下文:命令处理函数所需的共享依赖。"""

    run_program: Callable[[str], None]  # 执行完整 XQIASM 程序
    multi_line_input: Callable[..., str | None]  # 交互式多行输入


CommandHandler = Callable[[ShellContext, list[str]], None]


def cmd_ls(ctx: ShellContext, args: list[str]) -> None:
    """列出当前目录(目录 + .xqiasm/.txt 文件)。"""
    items = os.listdir(".")
    formatted = [
        f"<cbg>{i}/</cbg>" if os.path.isdir(i) else f"<cg>{i}</cg>"
        for i in items
        if os.path.isdir(i) or i.lower().endswith((".xqiasm", ".txt"))
    ]
    if formatted:
        _print(" ".join(formatted))


def cmd_begin(ctx: ShellContext, args: list[str]) -> None:
    """手动输入 XQI-BEGIN:进入交互式多行编辑模式。"""
    content = ctx.multi_line_input(initial_text="XQI-BEGIN\n")
    if content:
        ctx.run_program(content)


def cmd_execute_file(ctx: ShellContext, filename: str) -> None:
    """./xxx.XQIASM:读取并执行程序文件。"""
    filepath = os.path.abspath(filename)
    if not os.path.exists(filepath):
        _print(f'<cr>错误：文件</cr><cy2> {filepath} </cy2><cr>不存在</cr>')
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        ctx.run_program(f.read())


def cmd_mkdir(ctx: ShellContext, args: list[str]) -> None:
    folder_name = ' '.join(args)  # 兼容含空格名称(对齐原 split(' ',1) 语义)
    try:
        os.makedirs(folder_name, exist_ok=True)
        _print(f'<cg>文件夹已创建：</cg><cy2>{folder_name}</cy2>')
    except Exception as e:
        _print(f'<cr>创建失败：{str(e)}</cr>')


def cmd_rm(ctx: ShellContext, args: list[str]) -> None:
    target = ' '.join(args)
    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
            _print(f'<cg>文件夹已删除：</cg><cy2>{target}</cy2>')
        elif os.path.isfile(target):
            os.remove(target)
            _print(f'<cg>文件已删除：</cg><cy2>{target}</cy2>')
        else:
            _print(f'<cr>未找到目标：</cr><cy2>{target}</cy2>')
    except Exception as e:
        _print(f'<cr>删除失败：{str(e)}</cr>')


def cmd_cat(ctx: ShellContext, args: list[str]) -> None:
    filename = ' '.join(args)
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        _print(f'<ivory>{content}</ivory>')
    except Exception as e:
        _print(f'<cr>读取失败：{str(e)}</cr>')


def cmd_mv(ctx: ShellContext, args: list[str]) -> None:
    if len(args) != 2:
        _print('<cr>用法错误：mv 源文件 目标文件</cr>')
        return
    src, dst = args
    try:
        shutil.move(src, dst)
        _print(f'<cg>已移动/重命名：</cg><cy2>{src} → {dst}</cy2>')
    except Exception as e:
        _print(f'<cr>操作失败：{str(e)}</cr>')


def cmd_cp(ctx: ShellContext, args: list[str]) -> None:
    if len(args) != 2:
        _print('<cr>用法错误：cp 源文件 目标文件</cr>')
        return
    src, dst = args
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        _print(f'<cg>已复制：</cg><cy2>{src} → {dst}</cy2>')
    except Exception as e:
        _print(f'<cr>复制失败：{str(e)}</cr>')


def cmd_vim(ctx: ShellContext, args: list[str]) -> None:
    filename = ' '.join(args)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.xqiasm', '.txt'):
        print(f"不支持的文件类型: {ext}")
        return
    subprocess.run(["pyvim", filename])


def cmd_cd(ctx: ShellContext, args: list[str]) -> None:
    try:
        os.chdir(' '.join(args))
    except Exception as e:
        _print(f'<cr>切换失败：{str(e)}</cr>')


# 命令分发表:首词 → (处理函数, 最少参数个数;不足时静默忽略,对齐原行为)
COMMANDS: dict[str, tuple[CommandHandler, int]] = {
    "ls": (cmd_ls, 0),
    "XQI-BEGIN": (cmd_begin, 0),
    "mkdir": (cmd_mkdir, 1),
    "rm": (cmd_rm, 1),
    "cat": (cmd_cat, 1),
    "mv": (cmd_mv, 1),
    "cp": (cmd_cp, 1),
    "vim": (cmd_vim, 1),
    "cd": (cmd_cd, 1),
}


def dispatch(ctx: ShellContext, user_input: str) -> bool:
    """分发一行 shell 输入。

    返回 True 表示已处理;False 表示未识别(与历史行为一致,静默忽略)。
    './xxx.XQIASM' 形式优先于查表匹配。
    """
    if user_input.startswith('./') and user_input.endswith('.XQIASM'):
        cmd_execute_file(ctx, user_input[2:])
        return True

    parts = user_input.split()
    if not parts:
        return False
    entry = COMMANDS.get(parts[0])
    if entry is None:
        return False
    handler, min_args = entry
    if len(parts) - 1 < min_args:
        return False  # 原行为:不匹配任何分支,静默忽略
    handler(ctx, parts[1:])
    return True
