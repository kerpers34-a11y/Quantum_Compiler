import numpy as np
import pytest
from xqi.evaluator import QuantumEnvironment, Evaluator

# 一个假的 parser，用来避免 NoneType 报错
class DummyParser:
    def get_source_code_text(self, ast):
        return ""
    def get_labels_info(self, ast):
        return {}

def test_single_qubit_noise():
    env = QuantumEnvironment(qreg_size=1, simulation_mode='statevector')
    env.error_model = (1, 0.1)  # depolarizing noise
    env.quantum_state = np.array([1.0+0j, 0.0+0j])  # |0>
    env.apply_quantum_noise([0])
    # 检查结果仍然是 ndarray
    assert isinstance(env.quantum_state, np.ndarray)

def test_cnot_gate():
    env = QuantumEnvironment(qreg_size=2, simulation_mode='statevector')
    # 初始态 |10>
    env.quantum_state = np.array([0,0,1,0], dtype=np.complex128)
    dummy_parser = DummyParser()
    evaluator = Evaluator(env, parser=dummy_parser, ast=None)
    evaluator._apply_cnot_gate(control=0, target=1)
    # 检查是否变成 |11>
    expected = np.array([0,0,0,1], dtype=np.complex128)
    assert np.allclose(env.quantum_state, expected)

def test_measure_sampling():
    # 初始态 |+> = (|0> + |1>)/√2
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    probs = np.abs(state)**2  # [0.5, 0.5]
    outcomes = np.array([0, 1])
    samples = [np.random.choice(outcomes, p=probs) for _ in range(20)]
    # 检查采样结果只包含 0 和 1
    assert set(samples).issubset({0,1})
