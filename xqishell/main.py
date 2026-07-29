import os
import time

import pyperclip
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.key_binding import KeyBindings

from xqishell import (
    ASCII_ART_LOGO,
    CustomAutoSuggest,
    config,
    message_prompt,
    opcode_completer,
    style_html,
    style_prompt,
    xqiasm_lexer,
)
from xqishell.commands import ShellContext, dispatch
from xqishell.evaluator import Evaluator, QuantumEnvironment
from xqishell.parser import Parser
from xqishell.xqi_lexer import XQILexer

bindings = KeyBindings()

@bindings.add('c-c')
def _(event):
    event.app.exit()
@bindings.add('c-v')
def _(event):
    """从系统剪贴板读取内容并插入到当前光标位置(跨平台,经 pyperclip)"""
    data = event.app.clipboard.get_data()
    if data.text:
        event.current_buffer.insert_text(data.text)
        return
    try:
        text = pyperclip.paste()
    except Exception:
        return
    if text:
        event.current_buffer.insert_text(text.strip())

###################################################################################
###################################################################################

def write_debug_to_debug_file(_lexer):

    # 获取文件当前大小判断是否需要前置换行
    file_size = os.path.getsize(config.FILENAME_DEBUG)
    file_size_density_matrix = os.path.getsize(config.FILENAME_DEBUG_DENSITY_MATRIX)
    write_mode = 'a' if file_size > 0 else 'w'

    with open(config.FILENAME_DEBUG, mode=write_mode, encoding='utf-8') as f:
        # 非空文件时添加换行分隔
        if file_size > 0:
            f.write('\n')  # 添加分隔换行符

        # 写入自带换行格式的调试信息
        f.write(_lexer.debug_message)

        # 添加结束换行保证后续插入
        if not _lexer.debug_message.endswith('\n'):
            f.write('\n')

    with open(config.FILENAME_DEBUG_DENSITY_MATRIX, mode=write_mode, encoding='utf-8') as f:
        # 非空文件时添加换行分隔
        if file_size_density_matrix > 0:
            f.write('\n')  # 添加分隔换行符

        # 写入自带换行格式的调试信息
        f.write(_lexer.debug_message)

        # 添加结束换行保证后续插入
        if not _lexer.debug_message.endswith('\n'):
            f.write('\n')

###################################################################################
###################################################################################

def handle_multi_line_input(initial_text=""):
    """
    交互式多行输入模式。使用独立的 Session 避免 UI 污染。
    """
    print_formatted_text(HTML('<cg>进入程序编辑模式（在 XQI-END 后回车将执行并退出）</cg>'), style=style_html)

    multi_bindings = KeyBindings()

    @multi_bindings.add('enter')
    def _(event):
        buffer = event.app.current_buffer
        lines = buffer.text.split('\n')
        # 判定最后一行是否为结束符
        if lines and lines[-1].strip().upper() == 'XQI-END':
            event.app.exit(result=buffer.text)
        else:
            buffer.insert_text('\n')

    @multi_bindings.add('c-d')
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    @multi_bindings.add('c-v')
    def _(event):
        data = event.app.clipboard.get_data()
        event.current_buffer.insert_text(data.text)

    # 创建一个临时 Session
    temp_session = PromptSession(
        message=HTML('<ansiyellow>│</ansiyellow> '),
        prompt_continuation=HTML('<ansiyellow>│</ansiyellow> '),
        style=style_prompt,
        lexer=xqiasm_lexer,
        completer=opcode_completer,
        key_bindings=multi_bindings,
        auto_suggest=CustomAutoSuggest(),
        multiline=True,
        clipboard=PyperclipClipboard()
    )

    try:
        result = temp_session.prompt(default=initial_text)
        return result
    except KeyboardInterrupt:
        return None


###################################################################################
###################################################################################

###################################################################################
###################################################################################

def run_with_progress(user_input):
    """执行一段完整 XQIASM 程序(词法→语法→求值),并打印操作行数与调试信息。"""
    required_files = [config.FILENAME_DEBUG, config.FILENAME_DEBUG_DENSITY_MATRIX]
    for file in required_files:
        # 检查文件是否存在
        if not os.path.isfile(file):
            # 创建空文件
            with open(file, 'w'):
                print(f"Create file: {os.path.abspath(file)}")

    # 统计 XQI-BEGIN 与 XQI-END 之间的分号数(每条语句/注释一个,与原逐字符统计等价)
    begin = config.LENGTH_STRING_XQI_BEGIN
    end = len(user_input) - config.LENGTH_STRING_XQI_END
    operation_row_number = user_input[begin:end].count(';')

    print_formatted_text(HTML(f'<cbg>operation row number=</cbg><cbb>{operation_row_number}</cbb>'), style=style_html)
    lexer_main = XQILexer(user_input)
    write_debug_to_debug_file(lexer_main)
    parser = Parser(lexer_main)
    ast = parser.program()
    env = QuantumEnvironment()
    evaluator = Evaluator(env,parser,ast)
    evaluator.evaluate(ast)


# 兼容旧名(内部已统一使用 run_with_progress)
main_progress = run_with_progress

###################################################################################
###################################################################################

def _print_banner():
    """启动画面(保留原有的分步延迟打印)。"""
    print_formatted_text(HTML('<cg>###########################################</cg>'), style=style_html)
    print_formatted_text(HTML('<cbg>XQI: Quantum Computing Compiler and Simulator.</cbg>'), style=style_html)
    time.sleep(0.2)
    print_formatted_text(HTML('<cbg>version (debug)</cbg> <cbb>0.0.1</cbb>'), style=style_html)
    time.sleep(0.4)
    print_formatted_text(HTML('<cir>@Chengxian Deng. SCUT. 2020.</cir>'), style=style_html)
    time.sleep(0.5)
    print_formatted_text(ASCII_ART_LOGO)
    print_formatted_text(HTML('<cg>###########################################</cg>'), style=style_html)
    time.sleep(0.5)
    print_formatted_text(HTML('<cbg>Quantum Computing Compiler and Simulator will Execute!</cbg>'), style=style_html)
    time.sleep(0.6)
    print_formatted_text(HTML('<cg>XQI Shell is running...(按 ctrl+c 退出)</cg>'), style=style_html)
    time.sleep(0.8)


def _handle_program_block(ctx, raw_input):
    """处理包含 XQI-BEGIN 的输入:完整块直接执行,否则进入交互式多行编辑。"""
    if 'XQI-END' in raw_input.upper():
        # 情况 A：已经是完整的块（包含 BEGIN 和 END）
        print_formatted_text(HTML('<cg>检测到完整程序块，执行中...</cg>'), style=style_html)
        ctx.run_program(raw_input)
    else:
        # 情况 B：只有 BEGIN，进入交互式多行模式;补齐换行,带入已输入内容
        init_val = raw_input + ('\n' if not raw_input.endswith('\n') else '')
        content = ctx.multi_line_input(initial_text=init_val)
        if content:
            ctx.run_program(content)


def main():
    _print_banner()

    session = PromptSession(
        message_prompt,
        key_bindings=bindings,
        style=style_prompt,
        lexer=xqiasm_lexer,
        completer=opcode_completer,
        clipboard=PyperclipClipboard(),
        multiline=False,
        cursor=CursorShape.BLINKING_BEAM,
        wrap_lines=True,
    )
    ctx = ShellContext(run_program=run_with_progress, multi_line_input=handle_multi_line_input)

    while True:
        try:
            # 1. 获取输入
            raw_input = session.prompt()

            if raw_input is None: break
            if not raw_input.strip(): continue

            # 2. 程序块模式:只要包含 XQI-BEGIN 就进入代码块处理
            if 'XQI-BEGIN' in raw_input.upper():
                _handle_program_block(ctx, raw_input)
                print("")  # 打印空行，分隔输出与下一个 Prompt
                continue

            # 3. 常规指令模式 (只有不含 XQI-BEGIN 时才进入逐行解析)
            for current_line in raw_input.split('\n'):
                user_input = current_line.strip()
                if user_input:
                    dispatch(ctx, user_input)

        except KeyboardInterrupt:
            # 捕获主界面 Ctrl+C，不退出程序，只换行
            print("")
            continue
        except EOFError:
            break
        except Exception as e:
            print_formatted_text(HTML(f'<cr>系统错误：{str(e)}</cr>'), style=style_html)

if __name__ == '__main__':
    main()
