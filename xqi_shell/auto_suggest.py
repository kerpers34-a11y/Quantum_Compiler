from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

class CustomAutoSuggest(AutoSuggest):
    def get_suggestion(self, buffer, document):
        # 获取当前输入内容
        text = document.text

        # 定义建议规则（示例：匹配前缀）
        suggestions = {
            "U3": "U3(a,b,c) q[n]",
            "CNOT": "CNOT q[n],q[m]",
            "measure": "measure q[n]->c[m]",
            "GPS": "GPS(delta) q[n]",
            "MOV": "MOV R[n],R[m]",
            "ADD": "ADD R[d],R[n],R[m]",
            "SUB": "SUB R[d],R[n],R[m]",
            "MUL": "MUL R[d],R[n],R[m]",
            "DIV": "DIV R[d],R[n],R[m]",
            "LDR": "LDR R[n],M[m]",
            "STR": "STR R[n],M[m]",
            "CLDR": "CLDR c[n],M[m]",
            "CSTR": "CSTR c[n],M[m]",
            "shot": "shot 1",
            "error": "error(1)",
            "ERR": "ERR(1, R[n], R[m], 0.0, 0.0) q[a]",
            "qreg": "qreg q[n]",
            "creg": "creg c[n]",
            "reset": "reset q[n]",
            "barrier": "barrier q[n1], q[n2], ..."
        }

        # 查找匹配前缀的建议
        for prefix, suggestion in suggestions.items():
            if text.startswith(prefix) and len(text) > len(prefix):
                return Suggestion(suggestion[len(text):])
        return None