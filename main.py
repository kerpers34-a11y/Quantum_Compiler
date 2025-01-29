from rich import print
from register import *
import textprocessing_funs
from lexer import Lexer
from parser import Parser
"""
###################################################################################
###################################################################################
ONE_MATRIX: MATRIX = MATRIX()
Id_MATRIX: MATRIX = MATRIX()
ONE_Density_Matrix: MATRIX = MATRIX()
Id_Density_Matrix: MATRIX = MATRIX()

Measure_Event_Information: Measure_Event  = Measure_Event()
Measure_Density_Matrix_Event_Information: Measure_Density_Matrix_Event = Measure_Density_Matrix_Event()

nth_qubit: int = 0
nth_Classical: int = 0
###################################################################################
###################################################################################

input_filename_xqiasm = [None]*MAX_Operation_Column

print("###########################################\n")
print(">> [bold green]XQI: Quantum Computing Compiler and Simulator.[/bold green]\n")
print(">> [bold green]version (debug)[/bold green][bold blue] 0.0.1[/bold blue]\n")
print(">> [italic red]@Chengxian Deng. SCUT. 2020.[/italic red]\n")

print(textprocessing_funs.fetch_ascii_by_id(1))

print("###########################################\n")

print(">> [bold green]Quantum Computing Compiler and Simulator will Execute![/bold green]\n")
print(">> [bold green]Press[/bold green] [italic yellow]'Enter'[/italic yellow] [bold green]to Execute.[/bold green]\n")

textprocessing_funs.get_user_input()
print('\n')

input_filename_xqiasm[0] = '\n'
print(">> [bold green]Please Input Your XQI-ASM  File Name: [/bold green]")
filename_xqiasm=textprocessing_funs.get_user_input(prompt="Default: Quantum-PageRank-Two-Vertexes.XQIASM")
if filename_xqiasm=='':
    filename_xqiasm='Quantum-PageRank-Two-Vertexes.XQIASM'
    pass
print("\n>> [bold green]Quantum Computing Compiler and Simulator  is Running...[/bold green]\n\n")

# ONE_MATRIX.matrix_row = 1
# ONE_MATRIX.matrix_column = 1
ONE_MATRIX.matrix_element[0][0]=1

# Id_MATRIX.matrix_row = 2
# Id_MATRIX.matrix_column = 2
Id_MATRIX.matrix_element[0][0] = 1
Id_MATRIX.matrix_element[0][1] = 0
Id_MATRIX.matrix_element[1][0] = 0
Id_MATRIX.matrix_element[1][1] = 1

ONE_Density_Matrix.matrix_element=np.kron(ONE_MATRIX.matrix_element,np.conj(ONE_MATRIX.matrix_element).T)
Id_Density_Matrix=Id_MATRIX

Measure_Event_Information.Measure_Event_Quantum_Register_Bit = np.full(Quantum_Register_Number, False, dtype = bool)
Measure_Density_Matrix_Event_Information.Measure_Density_Matrix_Event_Quantum_Register_Bit = np.full(Quantum_Register_Number, False, dtype = bool)

with open(filename_xqiasm, 'r') as file:
    xqiasm_space = file.read()
xqiasm_space_size = len(xqiasm_space)
filename_debug = 'XQI-QC-list.txt'
filename_debug_matlab = 'XQI-QC-debug.dat'
filename_state_matlab = 'XQI-QC-state.dat'

filename_debug_Density_Matrix = 'XQI-QC-Density-Matrix-list.txt'
filename_debug_Density_Matrix_matlab = 'XQI-QC-Density-Matrix-debug.dat'
filename_Density_Matrix_state_matlab = 'XQI-QC-Density-Matrix-state.dat'

# 当成功读取到休止符时输出
# print(f"\n{filename_xqiasm}:\n")
# print(f"XQIASM source file space size: {xqiasm_space_size} Bytes.")
operation_row_number=0
for xqiasm_space_i_th in range(Length_string_XQI_BEGIN, xqiasm_space_size - Length_string_XQI_END):
    # 打印当前字符
    # print(xqiasm_space[xqiasm_space_i_th], end='')

    # 统计分号数量
    if xqiasm_space[xqiasm_space_i_th] == ';':
        operation_row_number += 1

# 打印统计结果
print(f"\n>> [bold green]operation row number={operation_row_number}[/bold green]\n")
"""

# 测试代码
code = """
XQI-BEGIN
ADD R[0], R[1], #5;
measure R[2]->R[3];
U3(0.233, R[3], 0.14567) q[1];
BNE end;
ADD R[4], R[5], #10;
end:
XQI-END
"""

lexer = Lexer(code)
parser = Parser(lexer)
ast = parser.program()
print(ast)