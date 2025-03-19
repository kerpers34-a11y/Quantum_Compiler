from rich import print
from register import *
import textprocessing_funs
from lexer import Lexer
from parser import Parser
from evaluator import QuantumEnvironment,Evaluator
#
# ###################################################################################
# ###################################################################################
#
#
# input_filename_xqiasm = [None]*MAX_Operation_Column
#
# print("###########################################\n")
# print(">> [bold green]XQI: Quantum Computing Compiler and Simulator.[/bold green]\n")
# print(">> [bold green]version (debug)[/bold green][bold blue] 0.0.1[/bold blue]\n")
# print(">> [italic red]@Chengxian Deng. SCUT. 2020.[/italic red]\n")
#
# print(textprocessing_funs.fetch_ascii_by_id(1))
#
# print("###########################################\n")
#
# print(">> [bold green]Quantum Computing Compiler and Simulator will Execute![/bold green]\n")
# print(">> [bold green]Press[/bold green] [italic yellow]'Enter'[/italic yellow] [bold green]to Execute.[/bold green]\n")
#
# textprocessing_funs.get_user_input()
# print('\n')
#
# input_filename_xqiasm[0] = '\n'
# print(">> [bold green]Please Input Your XQI-ASM  File Name: [/bold green]")
# filename_xqiasm=textprocessing_funs.get_user_input(prompt="Default: Quantum-PageRank-Two-Vertexes.XQIASM")
# if filename_xqiasm=='':
#     filename_xqiasm='Quantum-PageRank-Two-Vertexes.XQIASM'
#     pass
# print("\n>> [bold green]Quantum Computing Compiler and Simulator  is Running...[/bold green]\n\n")
#
# with open(filename_xqiasm, 'r') as file:
#     xqiasm_space = file.read()
# xqiasm_space_size = len(xqiasm_space)
# filename_debug = 'XQI-QC-list.txt'
# filename_debug_matlab = 'XQI-QC-debug.dat'
# filename_state_matlab = 'XQI-QC-state.dat'
#
# filename_debug_Density_Matrix = 'XQI-QC-Density-Matrix-list.txt'
# filename_debug_Density_Matrix_matlab = 'XQI-QC-Density-Matrix-debug.dat'
# filename_Density_Matrix_state_matlab = 'XQI-QC-Density-Matrix-state.dat'
#
# # 当成功读取到休止符时输出
# # print(f"\n{filename_xqiasm}:\n")
# # print(f"XQIASM source file space size: {xqiasm_space_size} Bytes.")
# operation_row_number=0
# for xqiasm_space_i_th in range(Length_string_XQI_BEGIN, xqiasm_space_size - Length_string_XQI_END):
#     # 打印当前字符
#     # print(xqiasm_space[xqiasm_space_i_th], end='')
#
#     # 统计分号数量
#     if xqiasm_space[xqiasm_space_i_th] == ';':
#         operation_row_number += 1
#
# # 打印统计结果
# print(f"\n>> [bold green]operation row number={operation_row_number}[/bold green]\n")
#
#
# lexer = Lexer(xqiasm_space)
# parser = Parser(lexer)
# ast = parser.program()
# print(ast)
# env = QuantumEnvironment(simulation_mode='statevector')
# evaluator = Evaluator(env)
# evaluator.evaluate(ast)

# # 测试代码
# code = """
# XQI-BEGIN
#  shot 1;
#  error(1);
#  qreg q[2]; define Quantum Register Number
#  creg c[2]; define Classical Register Number
#  MOV R[0],3.14159265;
#  MOV R[2],0.5236;
#  MOV R[3],1.570796325;
#  MOV R[4],4.712388975;
#  MOV R[10],0.00;
#  MOV R[11],0.00;
#  MOV R[12],0.00;
#  MOV R[15],5; loop for 5 time
#  MOV R[14],1;
#  debug;
#  loop: BL proc;
#  debug;
#  SUB R[15],R[15],R[14];
#  BNE loop;
#  B end;
#  proc: CNOT q[1],q[0];
#  U3(R[2],R[3],R[4]) q[0];
#  CNOT q[1],q[0];
#  U3(5.7595853,1.570796325,4.712388975) q[0];
#  U3(2.0944,1.570796325,4.712388975) q[0];
#  U3(0.0,1.570796325,1.570796325) q[0];
#  U3(4.1887853,1.570796325,4.712388975) q[0];
#  CNOT q[1],q[0];
#  U3(5.7595853,1.570796325,4.712388975) q[0];
#  CNOT q[1],q[0];
#  U3(0.5236,1.570796325,4.712388975) q[0];
#  ; SWAP
#  CNOT q[1],q[0];
#  CNOT q[0],q[1];
#  CNOT q[1],q[0];
#  MOV PC,LR;
#  end: measure q[0]->c[0];
#  measure q[1]->c[1];
#  MOV PC,0;
# XQI-END
# """
#
code="""XQI-BEGIN
   shot  1024;
   error(1);
  qreg q[2];  define Quantum Register Number
  creg c[2];  define Classical  Register Number
  ; period: SIX
  reset  q[0];
  reset  q[1];
  MOV R[0],3.14159265;
  MOV R[2],0.5236;
  MOV R[3],1.570796325;
  MOV R[4],4.712388975;
  MOV R[10],0.00;
  MOV R[11],0.00;
  MOV R[12],0.0;
  MOV R[15],5; loop for 5 time
  MOV R[14],1;
  ;debug;
  loop:  BL proc;
           debug;
           SUB R[15],R[15],R[14];
           BNE loop;
  B end;
  proc:  CNOT q[1],q[0]  ;
  U3(R[2],R[3],R[4]) q[0];
  CNOT q[1],q[0];
  U3(5.7595853,1.570796325,4.712388975) q[0];
  U3(2.0944,1.570796325,4.712388975) q[0];
  U3(0.0,1.570796325,1.570796325) q[0];
  U3(4.1887853,1.570796325,4.712388975) q[0];
  CNOT q[1],q[0];
  U3(5.7595853,1.570796325,4.712388975) q[0];
  CNOT q[1],q[0];
  U3(0.5236,1.570796325,4.712388975) q[0];
  ; SWAP
  CNOT q[1],q[0];
  CNOT q[0],q[1];
  CNOT q[1],q[0];
  MOV PC,LR;
end: measure q[0]->c[0];
 ; debug-p;
measure q[1]->c[1];
MOV PC,0;
XQI-END"""

lexer = Lexer(code)
parser = Parser(lexer)
ast = parser.program()
print(ast)
