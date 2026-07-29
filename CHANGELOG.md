# CHANGELOG

## Unreleased — 可读性/可维护性重构(2026-07)

本次重构以「先测试锁定行为,再小步重构」方式完成,全程 29 个测试守护
(黄金文件端到端 + bug 回归 + 对外兼容)。对外 import 路径、命令入口
(`xqishell = xqishell.main:main`)、config 旧常量名(全部保留 alias)均保持兼容。

### Bug 修复(10 个,各配回归测试)

- **B1**: `validate_gps_operands` 的 delta/qubit 两个检查写反,合法
  `GPS(delta) q[n];` 此前必报语法错误
- **B2**: `execute_rand` 误取 Opcode 节点切片,`rand` 指令此前调用即崩
- **B3**: `execute_gps` 在 statevector 模式下缩放的是密度矩阵别名而非
  state_vector(双模式下 GPS 此前实际上无效)
- **B4**: debug `.dat` 与 state `.dat` 字节序不一致,现统一为显式小端
  `'<ff'`(x86/ARM 主流平台原即为小端,既有文件字节内容不变)
- **B5**: 删除 REPL 中永不执行的第二个 `cat` 分支
- **B6**: 删除 error code 8/9 死分支(`execute_error` 只产生 code 1-6)
- **B7**: **新增实现 `CMP` / `BX` 指令**(此前词法声明但无法执行)。
  `CMP a,b` 计算 a-b 并置 ZF/SF(与 6 个条件分支约定一致);`BX LR`
  跳转至 LR(等价 `MOV PC,LR`)
- **B8**: `handle_err_instruction` 的 `@staticmethod` 带 `self` 怪胎改回普通方法
- **B9**: 修复 parser 死循环 —— 缺少 `error` 声明且含标签的程序此前会把
  解释器挂死,现在快速抛出语法错误
- **B10**: 修复词法遮蔽 —— `debug-p` 此前被 `debug` 前缀遮蔽,一用即报
  「非法字符: -」;指令交替串现按长度降序生成

### 结构改进

- `evaluator.py`(1355 行)拆为 `xqishell/evaluator/` 包:
  `environment`(量子态/噪声应用)、`noise`(Kraus 构造纯函数)、
  `executor`(指令执行)、`io`(文件输出 mixin)
- `main()` 上帝函数(约 180 行/15 个 elif)拆为 `commands.py` 命令分发表
- 新增单一数据源模块:`ast_nodes.py`(ASTNode 唯一定义)、
  `tokens.py`(Token NamedTuple)、`instructions.py`(34 条指令清单)、
  `errors.py`(XQIError 体系,多继承内置异常保兼容)
- `config.py` 规范化为 UPPER_SNAKE 命名,旧名全部保留 alias
- 删除约 20 处死代码;裸 except 全部改为精确异常类型;
  消除 import 即重置全局随机流的副作用(实例级 `default_rng`)
- `ascii_art.db`(单图 sqlite)移除,logo 改为 `style.py` 字符串常量;
  `setup.py`/`requirements.txt` 移除,构建配置统一为 `pyproject.toml`

### 行为注意点

- **随机流变化**: 测量采样从全局 `RandomState` 改为实例 `Generator`,
  同 seed 下序列与旧版不同(仅影响依赖旧随机序列的场景)
- **`rand` 与测量解耦**: `rand` 不再重置测量使用的随机流(行为改进)
- **`MOV SF/ZF, SF/ZF`**: 此前必 TypeError,现正确取值
- **性能**: Kraus 提升算符增加缓存,QPR2(shot 1024)运行时间 -42.6%
