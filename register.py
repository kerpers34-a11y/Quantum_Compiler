import numpy as np
from config import *
from dataclasses import dataclass,field
from functools import partial

"""
COMPLEX_dtype=np.dtype([('real', 'float'),('imag', 'float')])
def create_default_matrix():
    return np.empty((MAX_QUBITS_Corresponding_Code, MAX_QUBITS_Corresponding_Code), dtype=COMPLEX_dtype)
@dataclass
class MATRIX:
      matrix: np.ndarray = field(default_factory=create_default_matrix)
      matrix_row: int
      matrix_colum: int

class Quantum_Register_Classical_Register:
    Quantum_Register: MATRIX
    Classical_Register: MATRIX
"""

def create_default_matrix2(row,colum):
    return np.empty((row, colum))
def create_default_matrix1(value):
    return np.empty(value)

@dataclass
class MATRIX:
      matrix_element: np.ndarray = field(default_factory=partial(create_default_matrix2,MAX_QUBITS_Corresponding_Code,MAX_QUBITS_Corresponding_Code))
      # matrix_row: int = 0
      # matrix_colum: int = 0
@dataclass
class Quantum_Register_Classical_Register:
    Quantum_Register: MATRIX
    Classical_Register: MATRIX

@dataclass
class Measure_Event:
    Measure_Event_Classical_Register: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS))
    # corresponding probability.
    Measure_Event_Quantum_Register_Bit: np.ndarray = field(default_factory=partial(create_default_matrix1,MAX_QUBITS))
    # true: measured, false: not measured.
    Measure_Event_Quantum_Register: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS))
    # state after being project to |0> from initial state.

"""测试代码
b=MATRIX(matrix_row=1,matrix_colum=2)
print(b)
a=Measure_Event()
a.Measure_Event_Classical_Register = np.array([1, 0])
a.Measure_Event_Quantum_Register_Bit = np.array([1, 0])
a.Measure_Event_Quantum_Register = np.array([b, b])
print(a)
"""

@dataclass
class Complete_Measure_Event:
    resultant_measure_state_code: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # state code after all qubits measure.
    resultant_measure_state: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # state after all qubits measure.
    resultant_measured_state_probability: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # corresponding  measure probability.

@dataclass
class Measure_Density_Matrix_Event:
    Measure_Density_Matrix_Event_Q1_error_Probability: float = 0
    # Q1 error Probability parameter: measure error Probability.
    Measure_Density_Matrix_Event_Q2_error_Probability: float = 0
    # Q2 error Probability parameter: reserve measure parameter.
    Measure_Density_Matrix_Event_BOOL_Set_Error_Model: bytearray = field(default_factory=lambda: bytearray())
    # Set error model
    Measure_Density_Matrix_Event_Q_error_Code: bytearray = field(default_factory=lambda: bytearray())
    # error code
    Measure_Density_Matrix_Event_Quantum_Register_Bit: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS))
    # true: measured, false: not measured.
    Measure_Density_Matrix_Event_Classical_Register: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS))
    # corresponding  probability.
    Measure_Density_Matrix_Event_Quantum_Register: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS))
    # density matrix state after being project to |0> from initial state.

@dataclass
class Complete_Measure_Density_Matrix_Event:
    resultant_measure_density_matrix_state_code: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # density matrix state code after all qubits measure.
    resultant_measure_density_matrix_state: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # density matrix state after all qubits measure.
    resultant_measured_density_matrix_state_probability: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_QUBITS_Corresponding_Code))
    # corresponding  measure density matrix probability.

# Symbol Address of Instruction,Symbol Table
@dataclass
class Label_Symbol_Table:
    Instruction_Sequence_Number: int = 0
    Label_Symbol: np.ndarray = field(default_factory=partial(create_default_matrix1, MAX_Label_Character_Number))

@dataclass
class Register:
    Third_Byte: bytearray = field(default_factory=lambda: bytearray())
    Second_Byte: bytearray = field(default_factory=lambda: bytearray())
    First_Byte: bytearray = field(default_factory=lambda: bytearray())
    Zero_Byte:  bytearray = field(default_factory=lambda: bytearray())

@dataclass
class Classical_LDRSTR_Register_Memory:
    Classical_Register: MATRIX
    LDRSTR_matrix_row: int = 0
    # loaded/stored Classical Register Sequence Number

input_filename_xqiasm = np.empty(MAX_Operation_Column,dtype=np.str_)
qiasm_space_matrix = np.empty((MAX_Operation_Row,MAX_Operation_Column),dtype=np.str_)

Quantum_Register_Number = int()
Classical_Register_Number = int()

# Register (float) R[0],...,R[MAX_Register-1].
Register_Space = np.empty(MAX_Register,dtype=np.float64)
# Memory (float)   M[0],...,M[MAX_Memory-1]
Memory_Space = np.empty(MAX_Memory,dtype=np.float64)
# PC, LR, CPSR
Program_Counter = int()
Link_Register = int()

Current_Program_Status_Register = Register()

Measure_Event_Information = Measure_Event()
Complete_Measure_Event_Information = Complete_Measure_Event()
X_Complete_Measure_Event_Information = Complete_Measure_Event()

Measure_Density_Matrix_Event_Information = Measure_Density_Matrix_Event()
Complete_Measure_Density_Matrix_Event_Information = Complete_Measure_Density_Matrix_Event()
X_Complete_Measure_Density_Matrix_Event_Information = Complete_Measure_Density_Matrix_Event()

Code_Label_Symbol_Number = int(0)
COMPLEX_dtype=np.dtype([('real', 'float'),('imag', 'float')])
Code_Label_Symbol_Table = np.empty(MAX_Code_Label_Symbol_Number)

BOOL_MOV_PC_NEW_VALUE=False