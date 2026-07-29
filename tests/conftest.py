"""pytest 公共夹具与管线辅助。

golden 测试的设计要点:
- API 级调用 Lexer→Parser→Evaluator,不经过 prompt_toolkit 交互层;
- 所有产物文件(.dat/.txt)通过 chdir 收敛到 tmp_path,不污染仓库;
- 测量随机性通过固定 seed 锁定(evaluator 模块 import 时会执行一次
  np.random.seed() 随机化,因此必须在 evaluate 前重新 seed)。
"""

from __future__ import annotations

import io
import os
import re
import struct
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_SEED = 12345

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class PipelineResult:
    """一次 API 级管线运行的产物。"""

    stdout: str  # 已归一化(去 ANSI、LF 换行)
    evaluator: object
    workdir: Path


def normalize_stdout(text: str) -> str:
    """去除终端转义与平台换行差异,得到可快照的稳定文本。"""
    text = ANSI_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def run_xqiasm(source: str, workdir: Path, seed: int = GOLDEN_SEED) -> PipelineResult:
    """API 级跑通 Lexer→Parser→Evaluator,标准输出与产物文件均收敛到 workdir。"""
    from xqishell.evaluator import Evaluator, QuantumEnvironment
    from xqishell.parser import Parser
    from xqishell.xqi_lexer import XQILexer

    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        # 固定测量随机流;必须在 evaluate 前设置(import evaluator 时类体会随机化一次)
        np.random.seed(seed)
        buf = io.StringIO()
        with redirect_stdout(buf):
            lexer = XQILexer(source)
            parser = Parser(lexer)
            ast = parser.program()
            env = QuantumEnvironment()
            evaluator = Evaluator(env, parser, ast)
            evaluator.evaluate(ast)
    finally:
        os.chdir(old_cwd)
    return PipelineResult(stdout=normalize_stdout(buf.getvalue()), evaluator=evaluator, workdir=workdir)


@dataclass
class StateDat:
    """XQI-QC[-Density-Matrix]-state.dat 解析结果(小端 float32 复数流)。"""

    qreg_size: int
    shot_total: int
    states: np.ndarray  # complex128;[shots, dim] 或 [shots, dim, dim]
    counts: np.ndarray  # uint32;[dim]


def parse_state_dat(path: Path, matrix: bool = False) -> StateDat:
    """解析 state.dat(向量或密度矩阵两种格式,差异仅在 tag 长度与每 shot 元素数)。"""
    data = path.read_bytes()
    tag_len = 24 if matrix else 8
    qreg_size, shot_total = struct.unpack_from("<II", data, tag_len)
    dim = 1 << qreg_size
    n_complex = dim * dim if matrix else dim
    offset = tag_len + 8
    floats = np.frombuffer(data, dtype="<f4", count=shot_total * n_complex * 2, offset=offset)
    complex_vals = floats[0::2] + 1j * floats[1::2]
    if matrix:
        states = complex_vals.reshape(shot_total, dim, dim)
    else:
        states = complex_vals.reshape(shot_total, dim)
    offset += shot_total * n_complex * 8
    assert data[offset : offset + 8].rstrip(b"\x00") == b"COUNT", f"{path.name} 缺少 COUNT 尾块"
    counts = np.frombuffer(data, dtype="<u4", count=dim, offset=offset + 8)
    return StateDat(qreg_size, shot_total, states, counts)


@pytest.fixture(autouse=True)
def _fixed_seed():
    """每个测试默认固定 numpy 全局随机流(需要其他 seed 的测试可自行重设)。"""
    np.random.seed(GOLDEN_SEED)
    yield


@pytest.fixture()
def golden_dir() -> Path:
    return GOLDEN_DIR
