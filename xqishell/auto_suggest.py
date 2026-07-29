from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

from xqishell.instructions import INSTR_META

# 指令签名建议(单一数据源,覆盖全部指令);模块级构建一次,不再每次按键重建
SUGGESTIONS = {name: meta.operands for name, meta in INSTR_META.items()}

# 前缀匹配按键长降序,避免短前缀遮蔽(如 'B' 遮蔽 'BEQ'、'debug' 遮蔽 'debug-p')
_PREFIXES = sorted(SUGGESTIONS, key=len, reverse=True)


class CustomAutoSuggest(AutoSuggest):
    """基于指令签名的行内灰字建议:输入命中指令前缀后,补全签名剩余部分。"""

    def get_suggestion(self, buffer, document):
        text = document.text
        for prefix in _PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                return Suggestion(SUGGESTIONS[prefix][len(text):])
        return None
