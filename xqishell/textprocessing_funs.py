def ensure_xqi_tags(content):
    """确保内容包含正确的首尾标记"""
    lines = content.split('\n')

    # 寻找第一个非空行
    begin_index = next((i for i, line in enumerate(lines) if line.strip()), 0)
    # 寻找最后一个非空行
    end_index = next((i for i, line in reversed(list(enumerate(lines))) if line.strip()), len(lines) - 1)

    # 标记修正逻辑
    if begin_index < len(lines):
        if lines[begin_index].strip() != 'XQI-BEGIN':
            lines.insert(begin_index, 'XQI-BEGIN')
    else:
        lines.append('XQI-BEGIN')

    if end_index >= 0:
        if lines[end_index].strip() != 'XQI-END':
            lines.insert(end_index + 1, 'XQI-END')
    else:
        lines.append('XQI-END')

    # 重建内容保持原始格式
    processed = '\n'.join(lines).strip()
    return f"XQI-BEGIN\n{processed}\nXQI-END" if not processed else processed
