"""文本处理辅助函数。"""


def ensure_xqi_tags(content):
    """确保内容包含 XQI-BEGIN / XQI-END 首尾标记,缺失则补齐。

    语义(以 main.py 历史上实际使用的实现为准):
    - 只要没有任何一行(去空白后)等于 'XQI-BEGIN',就在最前面插入;
    - 'XQI-END' 同理,在末尾追加;
    - 返回值保证以换行结尾。
    """
    begin_marker = 'XQI-BEGIN'
    end_marker = 'XQI-END'

    lines = [line.rstrip('\r\n') for line in content.split('\n')]

    # 自动添加缺失标记
    if not any(line.strip() == begin_marker for line in lines):
        lines.insert(0, begin_marker)
    if not any(line.strip() == end_marker for line in lines):
        lines.append(end_marker)

    return '\n'.join(lines) + '\n'  # 保证结尾换行
