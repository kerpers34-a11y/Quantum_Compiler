#define  MAX_QUBITS                                                 8
#define  MAX_Classical_Register                                   8
#define  MAX_QUBITS_Corresponding_Code               256
# Visual Studio 2013:  MAX_QUBITS_Corresponding_Code must <=2^8=256,  Otherwise stack ERROR.
# Debug Win32

# maximum shot: The number of times the circuit is executed.
#define MAX_shot_TIMES                                          8192



# pi/2=1.570796325
# pi/3=1.04719755
# pi/4=0.7853981625
#

from typing import Final

MAX_QUBITS: Final=8
MAX_Classical_Register: Final=8
MAX_QUBITS_Corresponding_Code: Final=256
# Visual Studio 2013:  MAX_QUBITS_Corresponding_Code must <=2^8=256,  Otherwise stack ERROR.
# Debug Win32

# maximum shot: The number of times the circuit is executed.
MAX_shot_TIMES: Final=8192

pi: Final=3.14159265
half_pi: Final=1.570796325
Blank_Space: Final=32
MAX_Operation_Row: Final=0xFFFF
MAX_Operation_Column: Final=0xFFF
MAX_Numberical: Final=20
true: Final=1
false: Final=0
inf: Final=0x7FFFFFFF
real_number_precision: Final=1e-8

Length_string_XQI_BEGIN: Final=9
Length_string_XQI_END: Final=7

# maximum register
MAX_Register: Final=16
MAX_Memory: Final=32
MAX_Label_Character_Number: Final=64
MAX_Code_Label_Symbol_Number: Final=128
MAX_Call_Code_Symbol_Number: Final=128



default_Q_error_Code: Final=1
default_Q1_error_Probability: Final=0.002
default_Q2_error_Probability: Final=0.01

default_reset_error_Probability: Final=0.001

default_measure_error_Probability: Final=0.001

default_amp_damping_gamma: Final=0.001   ####################################################???
default_phase_damping_gamma: Final=0.001  ####################################################???






