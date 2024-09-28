import sys
import msvcrt
from PIL import Image
import sqlite3

# 显示输入框Waiting函数
def get_user_input(prompt="Waiting..."):
    # 设置灰色的文字颜色
    GREY = "\033[90m"
    RESET = "\033[0m"
    # 打印灰色提示文字
    sys.stdout.write(GREY + prompt + RESET)
    sys.stdout.flush()

    input_chars = []
    prompt_cleared = False

    while True:
        if msvcrt.kbhit():  # 检测是否有按键按下
            char = msvcrt.getch()  # 获取单个字符
            if char == b'\r':  # 检测到回车键，结束输入
                break

            if not prompt_cleared:
                # 清除提示文字
                sys.stdout.write('\r' + ' ' * len(prompt) + '\r')
                sys.stdout.flush()
                prompt_cleared = True

            # 将输入的字符保存并显示在控制台
            if char == b'\x08':  # 检测退格键
                if input_chars:
                    input_chars.pop()
                    sys.stdout.write('\b \b')  # 退格删除字符
            else:
                input_chars.append(char.decode('utf-8'))
                sys.stdout.write(char.decode('utf-8'))
            sys.stdout.flush()

    # 返回输入的内容
    return ''.join(input_chars)

# 将图像转换为ASCII字符的函数
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
    ascii_str = ''
    for pixel_value in img.getdata():
        ascii_str += ascii_chars[pixel_value // 25]

    # 将ASCII字符串分割为多行，每行对应图像的一行像素
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
    cursor.execute('''
        INSERT INTO ascii_art (art) VALUES (?)
    ''', (ascii_str,))

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

