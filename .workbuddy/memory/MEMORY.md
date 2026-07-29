# Quantum_Complier 项目长期记忆

## 项目概况
XQIASM 量子汇编的词法/递归下降解析/双模式(statevector+density matrix)模拟器 + REPL shell。
Python>=3.11,构建唯一来源 pyproject.toml(hatchling),入口 `xqishell = xqishell.main:main`。
2026-07 完成全量重构(见 CHANGELOG.md 与计划文件 toasty-forging-newton.md)。

## 结构约定(重构后)
- `xqishell/evaluator/` 包: environment(QuantumEnvironment) / noise(Kraus 纯函数) /
  executor(Evaluator) / io(StateIO mixin);旧 `from xqishell.evaluator import ...` 全兼容
- 单一数据源: ast_nodes.py(ASTNode dataclass) / tokens.py(Token NamedTuple) /
  instructions.py(34 条指令+INSTR_META) / errors.py(XQIError 体系,多继承内置异常)
- commands.py: REPL 命令分发表(ShellContext + dispatch)
- config.py: UPPER_SNAKE 规范名 + 全部旧名 alias(**不许删旧名**,兼容优先于命名规范)
- tests/: conftest(run_xqiasm API 级管线 + parse_state_dat) / golden(QPR2 shot 8 + npz/stdout 基线
  + generate_baseline.py) / test_*_bugs(B1-B10 @pytest.mark.bug) / test_compat / test_shell_smoke

## 行为语义要点(勿踩)
- XQIASM 程序强制要求 `shot` 与 `error` 声明;`error(0)` = 关噪声
- `MOV PC,0` = 结束当前 shot;双模式下 state_vector 与 density_matrix 必须锁步演化
  (execute_measure 两个都读),quantum_state 恒为 density_matrix 别名
- 测量随机流: env.rng(实例 default_rng);rand 指令用独立局部随机源(已解耦)
- .dat 二进制统一显式小端 '<ff';黄金测试故意不做字节级比对(数值 allclose)
- 黄金基线仅在行为确需变更时重建(tests/golden/generate_baseline.py),并在提交中说明

## 工程门禁
- 提交前: `pytest -q` 全绿 + `ruff check xqishell tests`(豁免仅 E501/E701/B904,只许收窄)
- venv: `.venv/Scripts/python.exe`;装包用清华镜像(默认 PyPI 易超时)

## 环境高危事项(必读!)
- **禁止 `git rm`**(含 --cached):本环境会清空整个目录!用 `git add -A` 记录删除;
  误删后 `git restore <dir>/` 可从索引恢复
- pip 安装可能掏空包(剩空壳目录)→ 故障时 shutil.rmtree 空壳后重装
- 同文件连续 Edit 易冲突,需串行并适时重读;脚本先写文件再执行(heredoc 三引号坑)
