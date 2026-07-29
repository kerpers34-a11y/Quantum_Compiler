"""XQIASM 全局配置常量(唯一来源)。

命名约定:新代码统一使用 UPPER_SNAKE_CASE 规范名;
C 移植遗留的大小写混用旧名全部保留为 alias(见文件末尾),保证外部 import 兼容。
"""

from typing import Final

# ---------------------------------------------------------------- 寄存器与内存上限
MAX_QUBITS: Final = 8
MAX_CLASSICAL_REGISTER: Final = 8
# 受 C 版(Visual Studio 2013)栈大小限制,必须 <= 2^8 = 256
MAX_QUBITS_CORRESPONDING_CODE: Final = 256
MAX_SHOT_TIMES: Final = 8192  # 最大 shot 数(电路重复执行次数)

MAX_REGISTER: Final = 16
MAX_MEMORY: Final = 32
MAX_LABEL_CHARACTER_NUMBER: Final = 64
MAX_CODE_LABEL_SYMBOL_NUMBER: Final = 128
MAX_CALL_CODE_SYMBOL_NUMBER: Final = 128

# ---------------------------------------------------------------- 数值常量(C 版兼容取值)
PI: Final = 3.14159265  # 与 C 版一致的截断精度,勿改为 math.pi
HALF_PI: Final = 1.570796325  # pi/2,同为截断精度
BLANK_SPACE: Final = 32
MAX_OPERATION_ROW: Final = 0xFFFF
MAX_OPERATION_COLUMN: Final = 0xFFF
MAX_NUMERICAL: Final = 20
TRUE: Final = True
FALSE: Final = False
INF: Final = 0x7FFFFFFF  # C 版 INT_MAX 语义,并非数学上的无穷大
REAL_NUMBER_PRECISION: Final = 1e-8

# XQI-BEGIN / XQI-END 标记的字符串长度
LENGTH_STRING_XQI_BEGIN: Final = 9
LENGTH_STRING_XQI_END: Final = 7

# ---------------------------------------------------------------- 默认噪声模型参数
DEFAULT_Q_ERROR_CODE: Final = 1
DEFAULT_Q1_ERROR_PROBABILITY: Final = 0.002
DEFAULT_Q2_ERROR_PROBABILITY: Final = 0.01

DEFAULT_RESET_ERROR_PROBABILITY: Final = 0.001
DEFAULT_MEASURE_ERROR_PROBABILITY: Final = 0.001

DEFAULT_AMP_DAMPING_GAMMA: Final = 0.001
DEFAULT_PHASE_DAMPING_GAMMA: Final = 0.001

DEFAULT_THERMAL_RELAXATION_ERROR_T1: Final = 50
DEFAULT_THERMAL_RELAXATION_ERROR_T2: Final = 70
DEFAULT_THERMAL_RELAXATION_ERROR_TGATE: Final = 0.1

DEFAULT_PAULI_X_ERROR_PROBABILITY: Final = 0.001
DEFAULT_PAULI_Y_ERROR_PROBABILITY: Final = 0.001
DEFAULT_PAULI_Z_ERROR_PROBABILITY: Final = 0.001

DEFAULT_COHERENT_X_UNITARY_ERROR_PROBABILITY: Final = 0
DEFAULT_COHERENT_Y_UNITARY_ERROR_PROBABILITY: Final = 0
DEFAULT_COHERENT_Z_UNITARY_ERROR_PROBABILITY: Final = 0.01

# ---------------------------------------------------------------- 输出文件名(相对 CWD)
FILENAME_DEBUG: Final = 'XQI-QC-list.txt'
FILENAME_DEBUG_MATLAB: Final = 'XQI-QC-debug.dat'
FILENAME_STATE_MATLAB: Final = 'XQI-QC-state.dat'
FILENAME_DEBUG_DENSITY_MATRIX: Final = 'XQI-QC-Density-Matrix-list.txt'
FILENAME_DEBUG_DENSITY_MATRIX_MATLAB: Final = 'XQI-QC-Density-Matrix-debug.dat'
FILENAME_DENSITY_MATRIX_STATE_MATLAB: Final = 'XQI-QC-Density-Matrix-state.dat'

# ---------------------------------------------------------------- 旧名 alias(仅为兼容保留,新代码勿用)
MAX_Classical_Register = MAX_CLASSICAL_REGISTER
MAX_QUBITS_Corresponding_Code = MAX_QUBITS_CORRESPONDING_CODE
MAX_shot_TIMES = MAX_SHOT_TIMES
MAX_Register = MAX_REGISTER
MAX_Memory = MAX_MEMORY
MAX_Label_Character_Number = MAX_LABEL_CHARACTER_NUMBER
MAX_Code_Label_Symbol_Number = MAX_CODE_LABEL_SYMBOL_NUMBER
MAX_Call_Code_Symbol_Number = MAX_CALL_CODE_SYMBOL_NUMBER

pi = PI
half_pi = HALF_PI
Blank_Space = BLANK_SPACE
MAX_Operation_Row = MAX_OPERATION_ROW
MAX_Operation_Column = MAX_OPERATION_COLUMN
MAX_Numberical = MAX_NUMERICAL  # 旧名拼写沿 C 版,不再修正以免破坏兼容
true = TRUE
false = FALSE
inf = INF
real_number_precision = REAL_NUMBER_PRECISION

Length_string_XQI_BEGIN = LENGTH_STRING_XQI_BEGIN
Length_string_XQI_END = LENGTH_STRING_XQI_END

default_Q_error_Code = DEFAULT_Q_ERROR_CODE
default_Q1_error_Probability = DEFAULT_Q1_ERROR_PROBABILITY
default_Q2_error_Probability = DEFAULT_Q2_ERROR_PROBABILITY
default_reset_error_Probability = DEFAULT_RESET_ERROR_PROBABILITY
default_measure_error_Probability = DEFAULT_MEASURE_ERROR_PROBABILITY
default_amp_damping_gamma = DEFAULT_AMP_DAMPING_GAMMA
default_phase_damping_gamma = DEFAULT_PHASE_DAMPING_GAMMA
default_thermal_relaxation_error_T1 = DEFAULT_THERMAL_RELAXATION_ERROR_T1
default_thermal_relaxation_error_T2 = DEFAULT_THERMAL_RELAXATION_ERROR_T2
default_thermal_relaxation_error_Tgate = DEFAULT_THERMAL_RELAXATION_ERROR_TGATE
default_pauli_X_error_Probability = DEFAULT_PAULI_X_ERROR_PROBABILITY
default_pauli_Y_error_Probability = DEFAULT_PAULI_Y_ERROR_PROBABILITY
default_pauli_Z_error_Probability = DEFAULT_PAULI_Z_ERROR_PROBABILITY
default_coherent_X_unitary_error_Probability = DEFAULT_COHERENT_X_UNITARY_ERROR_PROBABILITY
default_coherent_Y_unitary_error_Probability = DEFAULT_COHERENT_Y_UNITARY_ERROR_PROBABILITY
default_coherent_Z_unitary_error_Probability = DEFAULT_COHERENT_Z_UNITARY_ERROR_PROBABILITY

filename_debug = FILENAME_DEBUG
filename_debug_matlab = FILENAME_DEBUG_MATLAB
filename_state_matlab = FILENAME_STATE_MATLAB
filename_debug_Density_Matrix = FILENAME_DEBUG_DENSITY_MATRIX
filename_debug_Density_Matrix_matlab = FILENAME_DEBUG_DENSITY_MATRIX_MATLAB
filename_Density_Matrix_state_matlab = FILENAME_DENSITY_MATRIX_STATE_MATLAB
