"""重新生成黄金基线(qpr2_baseline.npz + qpr2_stdout.txt)。

用法:
    .venv/Scripts/python.exe tests/golden/generate_baseline.py

仅在「行为确需变更」的提交中运行(例如 rng 实例化替换全局随机流),
并在对应提交信息中明确说明基线重建原因。常规重构不应需要重建基线。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_DIR))  # 复用 conftest 中的管线辅助
sys.path.insert(0, str(PROJECT_ROOT))  # 导入 xqishell 包

from conftest import GOLDEN_SEED, parse_state_dat, run_xqiasm  # noqa: E402

GOLDEN = Path(__file__).resolve().parent


def main() -> None:
    source = (GOLDEN / "QPR2.XQIASM").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = run_xqiasm(source, tmp_path, seed=GOLDEN_SEED)
        sv = parse_state_dat(tmp_path / "XQI-QC-state.dat", matrix=False)
        dm = parse_state_dat(tmp_path / "XQI-QC-Density-Matrix-state.dat", matrix=True)

    (GOLDEN / "qpr2_stdout.txt").write_text(result.stdout, encoding="utf-8", newline="\n")
    np.savez(
        GOLDEN / "qpr2_baseline.npz",
        sv_states=sv.states,
        sv_counts=sv.counts,
        dm_states=dm.states,
        dm_counts=dm.counts,
    )
    print(f"baseline written to {GOLDEN}")
    print(f"  sv_states {sv.states.shape}  dm_states {dm.states.shape}")
    print(f"  sv_counts {sv.counts.tolist()}  dm_counts {dm.counts.tolist()}")


if __name__ == "__main__":
    main()
