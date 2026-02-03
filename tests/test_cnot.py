import numpy as np


def test_cnot():
    dim = 4

    # 初始态：把 q[0] 做成 |+⟩ = (|0⟩ + |1⟩)/√2，q[1]=|0⟩ → (|00⟩ + |10⟩)/√2
    state = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0], dtype=complex)
    rho = np.outer(state, state.conj())

    print("Initial diagonal:", [rho[k, k].real for k in range(4)])  # 应为 [0.5, 0, 0.5, 0]

    # 测试 control=0 (q[0]), target=1 (q[1])
    cnot_01 = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> 0) & 1:  # control q[0] == 1
            j = i ^ (1 << 1)  # 翻转 q[1]
            cnot_01[j, i] = 1.0
            cnot_01[i, i] = 0.0

    rho_01 = cnot_01 @ rho @ cnot_01.conj().T
    diag_01 = [rho_01[k, k].real for k in range(4)]
    off_01 = rho_01[0, 2].real  # |00⟩ 和 |11⟩ 之间的相干

    print("After CNOT control q[0] → target q[1] diagonal:", diag_01)
    print("Off-diagonal [0,2]:", off_01)  # 应为 0.5 或 -0.5

    # 测试 control=1 (q[1]), target=0 (q[0])
    # 为了测试这个方向，需要让 q[1] 有叠加，先做 |0+⟩
    state2 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], dtype=complex)  # (|00⟩ + |01⟩)/√2
    rho2 = np.outer(state2, state2.conj())

    cnot_10 = np.eye(dim, dtype=complex)
    for i in range(dim):
        if (i >> 1) & 1:  # control q[1] == 1
            j = i ^ (1 << 0)  # 翻转 q[0]
            cnot_10[j, i] = 1.0
            cnot_10[i, i] = 0.0

    rho_10 = cnot_10 @ rho2 @ cnot_10.conj().T
    diag_10 = [rho_10[k, k].real for k in range(4)]
    off_10 = rho_10[0, 1].real  # 调整观察相干位置

    print("After CNOT control q[1] → target q[0] diagonal:", diag_10)
    print("Off-diagonal example:", off_10)

test_cnot()