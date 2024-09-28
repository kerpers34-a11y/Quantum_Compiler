from rich import print
from register import *
import textprocessing_funs

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

BOOL_Input_File_Name = True

if BOOL_Input_File_Name:
    input_filename_xqiasm[0] = '\n'
    print(">> [bold green]Please Input Your XQI-ASM  File Name:   [/bold green]")
    pass
