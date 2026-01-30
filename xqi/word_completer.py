from prompt_toolkit.completion import WordCompleter

opcode_completer = WordCompleter(["XQI-BEGIN", "XQI-END", "shot", "error", "ERR", "U3", "measure", "CNOT", "CMP", "GPS", "MOV", "B", "BX", "BL", "BEQ", "BNE", "BGT", "BGE", "BLT", "BLE", "ADD", "SUB", "MUL", "DIV", "LDR", "STR", "CLDR", "CSTR", "qreg", "creg", "reset", "debug", "debug-p", "rand", "barrier"])
#(XQI\-BEGIN|XQI\-END|shot|error|ERR|U3|measure|CNOT|CMP|GPS|MOV|B|BX|BL|BEQ|BNE|BGT|BGE|BLT|BLE|ADD|SUB|MUL|DIV|LDR|STR|CLDR|CSTR|qreg|creg|reset|debug|debug\-p|rand|barrier)