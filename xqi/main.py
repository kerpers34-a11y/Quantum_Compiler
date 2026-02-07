import os
import time
import shutil
import subprocess

from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard

from xqi.xqi_lexer import XQILexer
from xqi.parser import Parser
from xqi.evaluator import QuantumEnvironment,Evaluator
from xqi import message_prompt, style_prompt, xqiasm_lexer, opcode_completer, style_html, CustomAutoSuggest, config, textprocessing_funs

bindings = KeyBindings()

@bindings.add('c-c')
def _(event):
    event.app.exit()
@bindings.add('c-v')
def _(event):
    """
    强制从系统剪贴板读取内容并插入到当前光标位置
    """
    # 尝试获取剪贴板内容
    data = event.app.clipboard.get_data()
    if data.text:
        event.current_buffer.insert_text(data.text)
    else:
        import subprocess
        try:
            text = subprocess.check_output(['powershell', '-command', 'Get-Clipboard'], encoding='utf-8')
            event.current_buffer.insert_text(text.strip())
        except:
            pass

###################################################################################
###################################################################################

def write_debug_to_debug_file(_lexer):

    # 获取文件当前大小判断是否需要前置换行
    file_size = os.path.getsize(config.filename_debug)
    file_size_density_matrix = os.path.getsize(config.filename_debug_Density_Matrix)
    write_mode = 'a' if file_size > 0 else 'w'

    with open(config.filename_debug, mode=write_mode, encoding='utf-8') as f:
        # 非空文件时添加换行分隔
        if file_size > 0:
            f.write('\n')  # 添加分隔换行符

        # 写入自带换行格式的调试信息
        f.write(_lexer.debug_message)

        # 添加结束换行保证后续插入
        if not _lexer.debug_message.endswith('\n'):
            f.write('\n')

    with open(config.filename_debug_Density_Matrix, mode=write_mode, encoding='utf-8') as f:
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



def ensure_xqi_tags(content):
    """增强版标记检测"""
    begin_marker = 'XQI-BEGIN'
    end_marker = 'XQI-END'

    lines = [line.rstrip('\r\n') for line in content.split('\n')]

    # 自动添加缺失标记
    if not any(line.strip() == begin_marker for line in lines):
        lines.insert(0, begin_marker)
    if not any(line.strip() == end_marker for line in lines):
        lines.append(end_marker)

    return '\n'.join(lines) + '\n'  # 保证结尾换行

###################################################################################
###################################################################################

def mini_vim(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.xqiasm', '.txt'):
        print(f"不支持的文件类型: {ext}")
        return

    subprocess.run(["pyvim", filename])

def view_text_file(filename):
    """统一处理 .xqiasm 和 .txt 文件内容查看"""
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ('.xqiasm', '.txt'):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            print_formatted_text(HTML(f'<ivory>{content}</ivory>'), style=style_html)
        else:
            print_formatted_text(HTML(f'<cr>不支持的文件类型：</cr><cy2>{ext}</cy2>'), style=style_html)
    except Exception as e:
        print_formatted_text(HTML(f'<cr>读取失败：{str(e)}</cr>'), style=style_html)

###################################################################################
###################################################################################

def handle_file_execution(filename):
    filepath = os.path.abspath(filename)  # 转成绝对路径
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), filepath
    else:
        print_formatted_text(HTML(f'<cr>错误：文件</cr><cy2> {filepath} </cy2><cr>不存在</cr>'), style=style_html)
        return None, None


###################################################################################
###################################################################################

def main_progress(user_input):
    required_files = [config.filename_debug, config.filename_debug_Density_Matrix]
    for file in required_files:
        # 检查文件是否存在
        if not os.path.isfile(file):
            # 创建空文件
            with open(file, 'w') as f:
                print(f"Create file: {os.path.abspath(file)}")
        else:
            pass

    xqiasm_space_size = len(user_input)
    operation_row_number = 0
    for xqiasm_space_i_th in range(config.Length_string_XQI_BEGIN, xqiasm_space_size - config.Length_string_XQI_END):
        # 打印当前字符
        # print(xqiasm_space[xqiasm_space_i_th], end='')

        # 统计分号数量
        if user_input[xqiasm_space_i_th] == ';':
            operation_row_number += 1

    print_formatted_text(HTML(f'<cbg>operation row number=</cbg><cbb>{operation_row_number}</cbb>'), style=style_html)
    lexer_main = XQILexer(user_input)
    write_debug_to_debug_file(lexer_main)
    parser = Parser(lexer_main)
    ast = parser.program()
    env = QuantumEnvironment()
    evaluator = Evaluator(env,parser,ast)
    evaluator.evaluate(ast)

###################################################################################
###################################################################################

def main():
    print_formatted_text(HTML('<cg>###########################################</cg>'), style=style_html)
    print_formatted_text(HTML('<cbg>XQI: Quantum Computing Compiler and Simulator.</cbg>'), style=style_html)
    time.sleep(0.2)
    print_formatted_text(HTML('<cbg>version (debug)</cbg> <cbb>0.0.1</cbb>'), style=style_html)
    time.sleep(0.4)
    print_formatted_text(HTML('<cir>@Chengxian Deng. SCUT. 2020.</cir>'), style=style_html)
    time.sleep(0.5)
    print_formatted_text(textprocessing_funs.fetch_ascii_by_id(1))
    print_formatted_text(HTML('<cg>###########################################</cg>'), style=style_html)
    time.sleep(0.5)
    print_formatted_text(HTML('<cbg>Quantum Computing Compiler and Simulator will Execute!</cbg>'), style=style_html)
    time.sleep(0.6)
    print_formatted_text(HTML('<cg>XQI Shell is running...(按 ctrl+c 退出)</cg>'), style=style_html)
    time.sleep(0.8)

    system_clipboard = PyperclipClipboard()

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

    while True:
        try:
            # 1. 获取输入
            raw_input = session.prompt()

            if raw_input is None: break
            if not raw_input.strip(): continue

            # 2. 预处理：判定是否包含代码块标记
            # 无论是首行还是中间包含，只要有 XQI-BEGIN 就启动代码块处理
            full_input_upper = raw_input.upper()

            if 'XQI-BEGIN' in full_input_upper:
                # 情况 A：已经是完整的块（包含 BEGIN 和 END）
                if 'XQI-END' in full_input_upper:
                    print_formatted_text(HTML('<cg>检测到完整程序块，执行中...</cg>'), style=style_html)
                    main_progress(raw_input)
                else:
                    # 情况 B：只有 BEGIN，进入交互式多行模式
                    # 补齐换行，带入已输入内容
                    init_val = raw_input + ('\n' if not raw_input.endswith('\n') else '')
                    content = handle_multi_line_input(initial_text=init_val)
                    if content:
                        main_progress(content)

                # 执行完程序块后，强制跳过本次循环剩余逻辑，回到 shell 顶层
                print("")  # 打印空行，分隔输出与下一个 Prompt
                continue

            # 3. 常规指令模式 (只有不含 XQI-BEGIN 时才进入逐行解析)
            raw_lines = raw_input.split('\n')
            for current_line in raw_lines:
                user_input = current_line.strip()
                if not user_input: continue
                # 指令分发逻辑
                if user_input == 'ls':
                    items = os.listdir(".")
                    formatted = [f"<cbg>{i}/</cbg>" if os.path.isdir(i) else f"<cg>{i}</cg>"
                                 for i in items if os.path.isdir(i) or i.lower().endswith((".xqiasm", ".txt"))]
                    if formatted: print_formatted_text(HTML(" ".join(formatted)), style=style_html)

                elif user_input == 'XQI-BEGIN':
                    # 这种情况属于手动输入 XQI-BEGIN，直接进入多行模式
                    content = handle_multi_line_input(initial_text="XQI-BEGIN\n")
                    if content: main_progress(content)

                elif user_input.startswith('./') and user_input.endswith('.XQIASM'):
                    content, filepath = handle_file_execution(user_input[2:])
                    if content: main_progress(content)
                elif '\n' in user_input:
                    filename = f"XQI_PASTED_{time.strftime('%Y%m%d_%H%M%S')}.XQIASM"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(user_input)

                    print_formatted_text(HTML(f'<cg>粘贴内容已保存：</cg><cy2>./{filename}</cy2>'), style=style_html)

                elif user_input.startswith('mkdir '):
                    # 创建文件夹
                    folder_name = user_input.split(' ', 1)[1]
                    try:
                        os.makedirs(folder_name, exist_ok=True)
                        print_formatted_text(HTML(f'<cg>文件夹已创建：</cg><cy2>{folder_name}</cy2>'),
                                             style=style_html)
                    except Exception as e:
                        print_formatted_text(HTML(f'<cr>创建失败：{str(e)}</cr>'), style=style_html)

                elif user_input.startswith('rm '):
                    # 删除文件或文件夹
                    target = user_input.split(' ', 1)[1]
                    try:
                        if os.path.isdir(target):
                            shutil.rmtree(target)
                            print_formatted_text(HTML(f'<cg>文件夹已删除：</cg><cy2>{target}</cy2>'),
                                                 style=style_html)
                        elif os.path.isfile(target):
                            os.remove(target)
                            print_formatted_text(HTML(f'<cg>文件已删除：</cg><cy2>{target}</cy2>'), style=style_html)
                        else:
                            print_formatted_text(HTML(f'<cr>未找到目标：</cr><cy2>{target}</cy2>'), style=style_html)
                    except Exception as e:
                        print_formatted_text(HTML(f'<cr>删除失败：{str(e)}</cr>'), style=style_html)

                elif user_input.startswith('cat '):
                    # 查看文件内容
                    filename = user_input.split(' ', 1)[1]
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            content = f.read()
                        print_formatted_text(HTML(f'<ivory>{content}</ivory>'), style=style_html)
                    except Exception as e:
                        print_formatted_text(HTML(f'<cr>读取失败：{str(e)}</cr>'), style=style_html)

                elif user_input.startswith('mv '):
                    # 移动或重命名文件
                    parts = user_input.split(' ')
                    if len(parts) == 3:
                        src, dst = parts[1], parts[2]
                        try:
                            shutil.move(src, dst)
                            print_formatted_text(HTML(f'<cg>已移动/重命名：</cg><cy2>{src} → {dst}</cy2>'),
                                                 style=style_html)
                        except Exception as e:
                            print_formatted_text(HTML(f'<cr>操作失败：{str(e)}</cr>'), style=style_html)
                    else:
                        print_formatted_text(HTML('<cr>用法错误：mv 源文件 目标文件</cr>'), style=style_html)

                elif user_input.startswith('cp '):
                    # 复制文件
                    parts = user_input.split(' ')
                    if len(parts) == 3:
                        src, dst = parts[1], parts[2]
                        try:
                            if os.path.isdir(src):
                                shutil.copytree(src, dst)
                            else:
                                shutil.copy2(src, dst)
                            print_formatted_text(HTML(f'<cg>已复制：</cg><cy2>{src} → {dst}</cy2>'), style=style_html)
                        except Exception as e:
                            print_formatted_text(HTML(f'<cr>复制失败：{str(e)}</cr>'), style=style_html)
                    else:
                        print_formatted_text(HTML('<cr>用法错误：cp 源文件 目标文件</cr>'), style=style_html)

                elif user_input.startswith('cat '):
                    filename = user_input.split(' ', 1)[1]
                    view_text_file(filename)


                elif user_input.startswith('vim '):
                    filename = user_input.split(' ', 1)[1]
                    mini_vim(filename)

                elif user_input.startswith('cd '):
                    folder = user_input.split(' ', 1)[1]
                    try:
                        os.chdir(folder)
                        # print_formatted_text(HTML(f'<cg>已切换目录：</cg><cy2>{os.getcwd()}</cy2>'),style=style_html)
                    except Exception as e:
                        print_formatted_text(HTML(f'<cr>切换失败：{str(e)}</cr>'), style=style_html)

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