from setuptools import setup, find_packages

setup(
    name="xqishell",                      # 包名
    version="0.1.0",                 # 版本号
    description="Quantum Computing Compiler and Simulator",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),        # 自动发现 xqishell/ 下的所有模块
    include_package_data=True,
    package_data={ "xqishell": ["ascii_art.db"],  },
    install_requires=[
        "markdown-it-py>=3.0,<4.0",
        "mdurl>=0.1,<1.0",
        "numpy>=2.1,<3.0",
        "prompt_toolkit>=3.0,<4.0",
        "pyvim>=3.0,<4.0",
        "Pygments>=2.18,<3.0",
        "scipy>=1.14,<2.0",
        "wcwidth>=0.2,<1.0",
        "pyperclip==1.11.0"
    ],
    entry_points={
        "console_scripts": [
            "xqishell = xqishell.main:main"   # 在终端提供 xqishell 命令，调用 main.py 中的 main()
        ]
    },
    python_requires=">=3.11",          # 根据你的环境指定最低 Python 版本
)
