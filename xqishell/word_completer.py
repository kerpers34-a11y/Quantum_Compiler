from prompt_toolkit.completion import WordCompleter

from xqishell.instructions import BLOCK_MARKERS, INSTRUCTIONS

# Tab 补全词表:块标记 + 全部指令(单一数据源)
opcode_completer = WordCompleter([*BLOCK_MARKERS, *INSTRUCTIONS])
