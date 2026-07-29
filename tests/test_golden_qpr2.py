"""黄金文件端到端测试:以 QPR2(shot 8)锁定 lexer→parser→evaluator 全链路行为。

基线生成于未修复任何 bug 的原始代码。重构期间必须保持:
- stdout 归一化快照精确一致;
- state/dm .dat 解析后数值 allclose(float32 精度,故意不做字节级比对,
  以便 B4 字节序修复不破坏本测试;字节级断言见 test_evaluator_bugs.py);
- 固定 seed 下 COUNT 数组精确一致。

若行为确需变更,运行 tests/golden/generate_baseline.py 重建基线并在提交中说明。
"""

from pathlib import Path

import numpy as np

from conftest import GOLDEN_SEED, parse_state_dat, run_xqiasm

GOLDEN = Path(__file__).parent / "golden"


def test_qpr2_golden(tmp_path):
    source = (GOLDEN / "QPR2.XQIASM").read_text(encoding="utf-8")
    result = run_xqiasm(source, tmp_path, seed=GOLDEN_SEED)

    baseline = np.load(GOLDEN / "qpr2_baseline.npz")

    # 1. stdout 归一化快照(精确一致)
    expected_stdout = (GOLDEN / "qpr2_stdout.txt").read_text(encoding="utf-8")
    assert result.stdout == expected_stdout

    # 2. 状态向量 .dat:逐 shot 数值 + COUNT
    sv = parse_state_dat(tmp_path / "XQI-QC-state.dat", matrix=False)
    assert sv.shot_total == 8
    np.testing.assert_allclose(sv.states, baseline["sv_states"], rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(sv.counts, baseline["sv_counts"])

    # 3. 密度矩阵 .dat:逐 shot 数值 + COUNT
    dm = parse_state_dat(tmp_path / "XQI-QC-Density-Matrix-state.dat", matrix=True)
    assert dm.shot_total == 8
    np.testing.assert_allclose(dm.states, baseline["dm_states"], rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(dm.counts, baseline["dm_counts"])
