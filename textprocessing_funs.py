from PIL import Image
import sqlite3
import time
import os
# 将图像转换为 ASCII 字符的函数
def image_to_ascii(image_path, width=80):
    # 打开图像并调整大小
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(aspect_ratio * width * 0.55)  # 保持宽高比
    img = img.resize((width, new_height))

    # 转换为灰度
    img = img.convert('L')

    # 定义 ASCII 字符列表
    ascii_chars = ['*', '*', '*', '*', '*', '*', '*', ':', ':', '.', '.']

    # 将像素值映射为 ASCII 字符
    ascii_str = ''.join(ascii_chars[pixel_value // 25] for pixel_value in img.getdata())

    # 将 ASCII 字符串分割为多行，每行对应图像的一行像素
    ascii_str = '\n'.join([ascii_str[i:(i + width)] for i in range(0, len(ascii_str), width)])
    print(ascii_str)
    return ascii_str


# 将 ASCII 字符串保存到 SQLite 数据库的函数
def save_ascii_to_db(ascii_str, db_name="ascii_art.db"):
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 创建表，如果不存在的话
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ascii_art (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        art TEXT NOT NULL
    )
    ''')

    # 插入 ASCII 字符串
    cursor.execute('INSERT INTO ascii_art (art) VALUES (?)', (ascii_str,))
    last_id = cursor.lastrowid

    # 提交事务并关闭连接
    conn.commit()
    conn.close()
    return last_id


# 从 SQLite 数据库中根据 ID 获取 ASCII 字符串
def fetch_ascii_by_id(ascii_id, db_name="ascii_art.db"):
    # 连接到 SQLite 数据库
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 查询特定 ID 对应的 ASCII 艺术
    cursor.execute('SELECT art FROM ascii_art WHERE id = ?', (ascii_id,))
    row = cursor.fetchone()

    # 关闭连接
    conn.close()

    # 如果存在该 ID，返回 ASCII 字符串，否则返回 None
    if row:
        return row[0]
    else:
        return None


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


def save_xqi_file(content):
    """智能保存文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"xqi_program_{timestamp}.XQIASM"

    with open(filename, 'w', encoding='utf-8') as f:
        final_content = ensure_xqi_tags(content)
        f.write(final_content)

    return filename

def list_xqi_files():
    """实现ls命令功能"""
    files = []
    for f in os.listdir('.'):
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.xqiasm', '.dat', '.txt') and f != 'requirements.txt':
                files.append(f)
    return sorted(files)