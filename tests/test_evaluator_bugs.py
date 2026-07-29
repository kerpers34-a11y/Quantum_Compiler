"""已实锤 bug 的回归测试(evaluator 侧)。

B2: execute_rand 误取 node.children[0](Opcode 'rand'),int('rand'[2:-1]) 即 int('n') 必崩;
B3: execute_gps 中 quantum_state 恒为 density_matrix 别名 —— statevector 分支缩放错对象,
    density 分支 phase*rho*phase.conj() 数学上是 no-op;
B4: debug .dat 用原生字节序 'ff',state .dat 用小端 '<ff',格式不自洽
    (x86 上字节恰好相同,故用 struct.pack 格式串探针做平台无关断言);
B6: error code 8/9 死分支(execute_error 只产生 1-6);
B7: CMP/BX 半成品 —— 词法声明但 evaluator 无 execute_cmp/execute_bx(本次补全实现)。

修复前全部为红。
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import PROJECT_ROOT, run_xqiasm


def wrap(body: str) -> str:
    """包装为合法 XQIASM 程序(parser 强制要求 shot 与 error 声明;error(0)=关闭噪声)。"""
    return f"XQI-BEGIN\nshot 1;\nerror(0);\nqreg q[1];\ncreg c[1];\n{body}\nMOV PC,0;\nXQI-END\n"


# ---------------------------------------------------------------- B2: rand
@pytest.mark.bug
def test_b2_rand_uses_operands_not_opcode(tmp_path):
    """rand R[0],R[1]; 以 R[1] 为种子向 R[0] 写入 [0,1) 均匀随机数(当前调用即崩)。"""
    result = run_xqiasm(wrap("MOV R[1],42;\nrand R[0],R[1];"), tmp_path)
    expected = np.random.default_rng(42).uniform(0, 1)
    assert result.evaluator.env.registers[0] == pytest.approx(expected)


# ---------------------------------------------------------------- B3: gps
@pytest.mark.bug
def test_b3_gps_applies_global_phase_to_statevector(tmp_path):
    """statevector 模式下 GPS(δ) 必须把 state_vector 缩放 e^{iδ}(当前缩的是密度矩阵别名)。"""
    result = run_xqiasm(wrap("GPS(0.5) q[0];"), tmp_path)
    sv = result.evaluator.env.state_vector
    np.testing.assert_allclose(sv, [np.exp(0.5j), 0.0], atol=1e-12)


# ---------------------------------------------------------------- B4: 字节序
@pytest.mark.bug
def test_b4_debug_dat_explicit_little_endian(tmp_path, monkeypatch):
    """debug .dat 的复数写入必须使用显式小端 '<ff',与 state .dat 一致。"""
    formats = []
    real_pack = struct.pack

    def spy(fmt, *args):
        formats.append(fmt)
        return real_pack(fmt, *args)

    monkeypatch.setattr("xqishell.evaluator.struct.pack", spy)
    run_xqiasm(wrap("debug;"), tmp_path)

    float_fmts = {f for f in formats if isinstance(f, str) and f.endswith("ff")}
    assert float_fmts, "未捕获到任何复数(float,float)写入"
    assert all(f.startswith("<") for f in float_fmts), f"存在非显式小端格式: {float_fmts}"


# ---------------------------------------------------------------- B6: 死分支
@pytest.mark.bug
def test_b6_no_dead_error_code_branches():
    """源码中不应存在 error code 8/9 的死分支(execute_error 只产生 1-6)。"""
    joined = "\n".join(
        p.read_text(encoding="utf-8") for p in (PROJECT_ROOT / "xqishell").rglob("*.py")
    )
    assert "error_model[0] == 9" not in joined, "readout-error(code 9)死分支仍在"
    assert "code == 8" not in joined, "reset-error(code 8)死分支仍在"


# ---------------------------------------------------------------- B7: CMP/BX
CMP_EQ_PROG = """XQI-BEGIN
shot 1;
error(0);
qreg q[1];
creg c[1];
MOV R[0],5;
MOV R[1],5;
CMP R[0],R[1];
BEQ equal;
MOV R[2],0;
B done;
equal: MOV R[2],1;
done: MOV PC,0;
XQI-END
"""

CMP_LT_PROG = """XQI-BEGIN
shot 1;
error(0);
qreg q[1];
creg c[1];
MOV R[0],3;
MOV R[1],5;
CMP R[0],R[1];
BLT less;
MOV R[2],0;
B done;
less: MOV R[2],1;
done: MOV PC,0;
XQI-END
"""

BX_PROG = """XQI-BEGIN
shot 1;
error(0);
qreg q[1];
creg c[1];
BL sub;
B done;
sub: MOV R[0],7;
BX LR;
done: MOV R[1],1;
MOV PC,0;
XQI-END
"""


@pytest.mark.bug
def test_b7_cmp_equal_takes_beq(tmp_path):
    """CMP 相等 → ZF=1 → BEQ 跳转(当前 CMP 落入分支处理器报错)。"""
    result = run_xqiasm(CMP_EQ_PROG, tmp_path)
    assert result.evaluator.env.registers[2] == 1


@pytest.mark.bug
def test_b7_cmp_less_takes_blt(tmp_path):
    """CMP 有符号小于 → SF=1,ZF=0 → BLT 跳转。"""
    result = run_xqiasm(CMP_LT_PROG, tmp_path)
    assert result.evaluator.env.registers[2] == 1


@pytest.mark.bug
def test_b7_bx_lr_returns_to_caller(tmp_path):
    """BX LR 等价 MOV PC,LR:返回到 BL 下一条(当前 BX 落入分支处理器报错)。"""
    result = run_xqiasm(BX_PROG, tmp_path)
    assert result.evaluator.env.registers[0] == 7
    assert result.evaluator.env.registers[1] == 1


# ---------------------------------------------------------------- B10: debug-p
@pytest.mark.bug
def test_b10_debug_p_full_pipeline(tmp_path, monkeypatch):
    """debug-p 全链路可执行(词法遮蔽修复后);input 暂停行为打桩屏蔽。"""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    result = run_xqiasm(wrap("debug-p;"), tmp_path)
    assert "!SUCCESS!" in result.stdout
