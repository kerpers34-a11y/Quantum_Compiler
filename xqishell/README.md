# XQI-QC Shell 简介

## 功能特点

- **模拟终端**: 模拟linux终端风格，实现linux终端常用指令，使用方便
- **快捷执行**: 将(.XQIASM)文件设为可执行文件，./test.XQIASM 回车一键执行
- **完善提示**: 完善的XQI代码提示功能，方便使用
- **多行交互设计**: 终端输入XQI-BEGIN回车自动开启多行输入模式，调试方便
- **多误差模型支持**: 支持调节解极化误差、幅度/相位误差等6种误差模型

## 快速开始
### (非必要)
推荐下载使用 [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 隔离环境
```bash
conda create -n xqishell python=3.11 -y
conda activate xqishell
```
> （或使用 [venv](https://docs.python.org/3/library/venv.html) ）

```bash
python -m venv xqishell
source xqishell/bin/activate    # Windows 用 xqishell\Scripts\activate
```
### 下载
```bash
pip install xqishell
```
输入xqishell以唤醒终端 (ctrl+c以退出)
```bash
xqishell
```

## Shell 指令列表

| 命令名称  | 格式示例 | 功能说明 | 备注 |
|------|------|------|------|
|ls|ls|列出当前目录下所有 .xqiasm / .txt 文件和文件夹（带颜色区分）|无参数|
|XQI-BEGIN|XQI-BEGIN|手动触发多行程序编辑模式（出现 XQI-BEGIN 后进入多行输入，直到 XQI-END）|单独输入此行会启动编辑器模式|
|./文件名.XQIASM|./demo.XQIASM|直接执行当前目录下的 .XQIASM 文件（相对路径，需以 ./ 开头）| 只支持当前目录下的文件|
|mkdir|mkdir 新文件夹名|创建文件夹（支持不存在的父目录）|支持一个参数|
|rm|rm 要删除的文件或文件夹|删除文件 或 递归删除文件夹|支持一个参数（文件或目录）|
|cat|cat filename.xqiasm|显示文件内容（.xqiasm / .txt 优先使用高亮，其它文件用普通方式显示）|支持一个参数（文件名）|
|mv|mv oldname newname|移动文件/文件夹 或 重命名|必须正好两个参数（源 → 目标）|
|cp|cp source dest|复制文件 或 递归复制文件夹|必须正好两个参数（源 → 目标）|
|vim|vim filename.xqiasm|使用 pyvim 打开指定文件进行编辑（目前仅限 .xqiasm / .txt）,支持一个参数（文件名）|
|cd|cd 子目录名  ..|切换当前工作目录|支持一个参数（路径）|
## XQI-QC 常用汇编指令
- **shot**
定义shot次数
- **qreg**
定义量子寄存器的大小（例如 qreg q[5];），会重置量子系统并分配对应维度的状态向量和密度矩阵
- **creg**
定义经典寄存器的大小（例如 creg c[8];），初始化经典寄存器数组（用于存储测量结果）
- **error**
设置/关闭/修改全局错误模型，支持多种噪声类型（去极化、幅度衰减、相位衰减、热弛豫、Pauli、相干误差等），可指定不同概率参数
- **ERR**
局部误差指令（ERR(model,code,p1,p2,p3) q[...];），立即对指定量子位施加指定的Kraus噪声算符（独立于全局 error 模型）
- **MOV**
数据移动指令，支持：
• 寄存器 ↔ 寄存器
• 立即数 → 寄存器
• PC/LR/SF/ZF 的读写
• MOV PC, 0 可用于提前结束当前 shot
- **ADD / SUB / MUL / DIV**
寄存器间的二元浮点运算（R[i] ← R[j] ±/*÷ R[k] 或立即数），结果写入目标寄存器并更新 SF（符号标志）和 ZF（零标志）
- **U3**
通用单量子比特门（U3(θ,φ,λ)），最常用的参数化旋转门，支持立即数或寄存器参数
- **CNOT**
控制非门（受控-X），标准两量子比特纠缠门（control, target）
- **GPS**
Global Phase Shift，全局相位门，对整个量子态乘以 e^(iδ)，δ 可来自立即数或寄存器
- **measure**
测量指定量子比特到经典寄存器，支持理想采样（statevector）和带读出误差的物理采样（density matrix），会引起态坍缩
- **reset**
重置指定量子比特到 |0⟩，支持带重置误差的物理实现（根据 error model 中的 p_reset）
- **B / BL**
无条件跳转（Branch）和带链接寄存器的跳转（Branch with Link），用于实现函数调用与返回
- **BEQ / BNE / BGT / BGE / BLT / BLE**
条件分支指令，根据 ZF（零标志）和 SF（符号标志）判断是否跳转
- **LDR**
从内存加载浮点数到通用寄存器 R[]（Load Register）
- **STR**
将通用寄存器 R[] 的值存储到内存 M[]（Store Register）
- **CLDR**
从内存的实部+虚部两个位置加载复数值到经典寄存器 c[]（Complex Load Register）
- **CSTR**
将经典寄存器 c[] 的复数值拆成实部+虚部分别存入内存两个位置（Complex Store Register）
- **debug / debug-p**
调试输出指令，把当前量子态（statevector + density matrix）、寄存器、标志位、内存等信息写入文本和二进制调试文件；带 -p 的版本会暂停等待输入
- **barrier**
屏障指令，目前在模拟器中为空操作（主要用于语义分隔和未来可能的优化/调度）
- **rand**
产生 [0,1) 均匀随机数，写入目标寄存器 R[]，随机种子可由另一个寄存器控制


| 指令名       | 参数格式说明                                                                 | 功能描述 |
|--------------|-------------------------------------------------------------------------------|----------|
| **shot**     | `shot <N>` <br>其中 `<N>` 为整数，表示执行的次数（shots 数量）                 | 设置程序运行的总 shot 次数 |
| **qreg**     | `qreg q[<n>]` <br>其中 `<n>` 为整数，表示量子寄存器大小                        | 初始化量子寄存器大小 |
| **creg**     | `creg c[<n>]` <br>其中 `<n>` 为整数，表示经典寄存器大小                        | 初始化经典寄存器大小 |
| **error**    | `error <enable>, <code>, <params...>` <br>其中 `<enable>` 为 TRUE/FALSE 或 1/0 <br>`<code>` 为误差模型编号 <br>`<params...>` 为不同模型对应的参数 | 设置全局误差模型 |
| **ERR**      | `ERR(<model>, <code>, p1, p2, p3, ...) q[n];` <br>其中 `<model>` 为误差类型编号 <br>`<code>` 为误差代码 <br>`p1, p2, p3...` 为物理参数 <br>`q[n]` 为目标量子位 | 应用局部误差模型到指定量子位 |
| **U3**       | `U3(theta, phi, lambda) q[n]` <br>其中参数为浮点数，`q[n]` 为目标量子位        | 单比特通用旋转门 |
| **CNOT**     | `CNOT q[a], q[b]` <br>`q[a]` 为控制位，`q[b]` 为目标位                         | 双比特受控非门 |
| **GPS**      | `GPS <params> q[n]` <br>参数依赖实现，`q[n]` 为目标量子位                      | 特殊量子门（实现依赖） |
| **reset**    | `reset q[n]` <br>`q[n]` 为目标量子位                                          | 将目标量子位重置为 |0⟩ |
| **measure**  | `measure q[n] -> c[m]` <br>`q[n]` 为量子位，`c[m]` 为经典寄存器位              | 测量量子位并存入经典寄存器 |
| **barrier**  | `barrier`                                                                    | 插入屏障，阻止门优化跨越 |
| **debug**    | `debug`                                                                      | 调试指令，打印当前状态 |
| **debug-p**  | `debug-p`                                                                    | 调试指令，打印概率分布 |
| **Label**    | `<label>:`                                                                   | 定义标签，用于跳转指令 |
| **B/BL**     | `B <label>` 或 `BL <label>`                                                  | 无条件跳转（B），带链接寄存器保存的跳转（BL） |
| **BEQ/BNE**  | `BEQ <label>` 或 `BNE <label>`                                               | 条件跳转：等于/不等于 |
| **BGT/BGE**  | `BGT <label>` 或 `BGE <label>`                                               | 条件跳转：大于/大于等于 |
| **BLT/BLE**  | `BLT <label>` 或 `BLE <label>`                                               | 条件跳转：小于/小于等于 |
| **MOV**      | `MOV <dest>, <src>` <br>其中 `<dest>` 可为寄存器（PC, LR, SF, ZF）或普通寄存器 | 将源值赋给目标寄存器 |
| **ADD/SUB**  | `ADD <dest>, <src1>, <src2>` <br>`SUB <dest>, <src1>, <src2>`                | 整数加法/减法 |
| **MUL/DIV**  | `MUL <dest>, <src1>, <src2>` <br>`DIV <dest>, <src1>, <src2>`                | 整数乘法/除法 |

github链接：https://github.com/kerpers34-a11y/Quantum_Compiler

**祝您使用愉快！**