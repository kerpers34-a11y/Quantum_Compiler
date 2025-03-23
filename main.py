import os
import time

from pygments.lexer import Lexer
import config
import textprocessing_funs

from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.cursor_shapes import CursorShape

from xqi_lexer import XQILexer
from parser import Parser
from evaluator import QuantumEnvironment,Evaluator
from xqi_shell import message_prompt, style_prompt, xqiasm_lexer, opcode_completer, style_html, CustomAutoSuggest

bindings = KeyBindings()

@bindings.add('c-c')
def _(event):
    event.app.exit()

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

def handle_multi_line_input(session):
    """多行输入处理"""
    print_formatted_text(HTML('<cg>进入多行模式（输入XQI-END或按CTRL+D退出）</cg>'), style=style_html)
    multi_lines = []

    # 提示符函数

    def get_prompt(width=0, line_number=0, wrap_count=0):
        return HTML('<ansiyellow>│</ansiyellow> ')

        # 键绑定
    multi_bindings = KeyBindings()
    @ multi_bindings.add('c-d')
    def _(event):
        """退出多行输入"""
        buffer = event.app.current_buffer
        if buffer.text.strip():  # 如果当前缓冲区有内容
            multi_lines.append(buffer.text)  # 保存缓冲区内容
        event.app.exit(result=False)  # 退出多行模式

    def collect_input(event):
        """每次用户输入时保存当前行"""
        buffer = event.app.current_buffer
        # 获取当前缓冲区的所有行
        lines = buffer.text.splitlines()  # 将缓冲区内容按行切割
        if lines:
            current_line = lines[-1].strip()  # 仅收集当前输入行（最新的那一行）
            if current_line:
                multi_lines.append(current_line)  # 将当前行保存到 multi_lines 列表中
            # 检查当前行是否为 'XQI-END'，如果是，则退出多行模式
            if current_line == 'XQI-END':
                event.app.exit(result=False)  # 退出多行模式

    # 保持默认回车行为的同时，继续收集输入
    @multi_bindings.add('enter')
    def _(event):
        collect_input(event)  # 保存当前输入
        event.app.current_buffer.insert_text('\n')

    try:
        # 仅调用一次 session.prompt 来接收多行输入
        session.prompt(
            message=get_prompt(),
            prompt_continuation=get_prompt,
            multiline=True,
            key_bindings=multi_bindings,
            vi_mode=False
        )

        return '\n'.join(multi_lines)

    except KeyboardInterrupt:
        print_formatted_text(HTML('<cy>输入已中断</cy>'), style=style_html)
        return None
    except Exception as e:
        print_formatted_text(HTML(f'<cr>输入错误：</cr><cy2>{str(e)}<cy2>'), style=style_html)
        return None


def ensure_xqi_tags(content):
    """增强版标记检测"""
    begin_marker = 'XQI-BEGIN'
    end_marker = 'XQI-END'

    # 保留原始换行格式
    lines = [line.rstrip('\r\n') for line in content.split('\n')]

    # 自动添加缺失标记
    if not any(line.strip() == begin_marker for line in lines):
        lines.insert(0, begin_marker)
    if not any(line.strip() == end_marker for line in lines):
        lines.append(end_marker)

    return '\n'.join(lines) + '\n'  # 保证结尾换行

###################################################################################
###################################################################################

def handle_file_execution(filename):
    """处理文件执行命令"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print_formatted_text(HTML(f'<cr>错误：文件</cr><cy2> {filename} </cy2><cr>不存在</cr>'), style=style_html)
        return None

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
    # env = QuantumEnvironment(simulation_mode='statevector')
    env = QuantumEnvironment(simulation_mode='density_matrix')
    evaluator = Evaluator(env)
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

    while True:
        try:
            session = PromptSession(
                message_prompt,
                key_bindings=bindings,
                style=style_prompt,
                lexer=xqiasm_lexer,
                completer=opcode_completer,
                cursor=CursorShape.BLINKING_BEAM,
                auto_suggest=CustomAutoSuggest(),
                multiline=False,
                wrap_lines=True,
                enable_open_in_editor=True
            )
            user_input = session.prompt()

            if user_input is None:  # 处理直接按Ctrl+C的情况
                break

            # 命令分发逻辑
            if user_input.strip() == 'ls':
                files = textprocessing_funs.list_xqi_files()
                if files:
                    print_formatted_text(HTML('<cg>|</cg > ' + ' <cg></cg > '.join(files)), style=style_html)
                else:
                    print_formatted_text(HTML('<cr>未找到匹配文件</cr>'), style=style_html)

            elif user_input.strip() == 'XQI-BEGIN':

                content = handle_multi_line_input(session)

                if content:
                    processed_content = ensure_xqi_tags(content)
                    saved_file = textprocessing_funs.save_xqi_file(processed_content)
                    print_formatted_text(HTML(f'<cg>程序已保存至：</cg><cy2>./{saved_file}</cy2>'), style=style_html)
                    print_formatted_text(HTML(f'<cg>自动执行: </cg><cy2>./{saved_file}</cy2>'), style=style_html)
                    user_input = processed_content
                    main_progress(user_input)
                else:
                    print_formatted_text(HTML('<cr>未保存内容，返回单行模式</cr>'), style=style_html)

            elif user_input.startswith('./') and user_input.endswith('.XQIASM'):
                content = handle_file_execution(user_input[2:])
                if content:
                    user_input = content
                    print_formatted_text(HTML(f'<cg>已加载文件：</cg><ivory>{user_input[0:]}</ivory>'), style=style_html)
                    print_formatted_text(HTML(f'<cg>等待执行...</cg>'), style=style_html)
                    main_progress(user_input)

            elif '\n' in user_input:
                filename = f"XQI_PASTED_{time.strftime('%Y%m%d_%H%M%S')}.XQIASM"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(user_input)

                print_formatted_text(HTML(f'<cg>粘贴内容已保存：</cg><cy2>./{filename}</cy2>'), style=style_html)

        except KeyboardInterrupt:
            print_formatted_text(HTML('<cr>操作已中断</cr>'), style=style_html)
        except Exception as e:
            print_formatted_text(HTML(f'<cr>系统错误：{str(e)}</cr>'), style=style_html)

if __name__ == '__main__':
    main()