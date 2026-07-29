"""状态与调试文件输出(StateIO 混入)。

负责: state/density-matrix 的 .dat 二进制(显式小端)与调试 .txt/.dat 写入,
以及最终控制台报告(测量事件/统计计数)。
由 evaluator.py 拆包而来,以 mixin 形式并入 Evaluator,调用点不变。
"""

import os
import struct

import numpy as np

from xqishell import config


class StateIO:
    def _prepare_final_state_files(self):
        """【开头】创建文件，写入对齐的文件头"""
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()
        self.state_dat_path = os.path.join(log_dir, "XQI-QC-state.dat")
        self.dm_state_dat_path = os.path.join(log_dir, "XQI-QC-Density-Matrix-state.dat")

        # 1. 状态矢量文件初始化
        with open(self.state_dat_path, "wb") as f:
            # Tag: 'XQI-QC' (6 bytes) + padding (2 bytes) = 8 bytes
            f.write(b'XQI-QC'.ljust(8, b'\x00'))
            # <I 表示小端序 unsigned int (4字节)
            f.write(struct.pack('<I', self.env.qreg_size))
            f.write(struct.pack('<I', self.shot_total))

        # 2. 密度矩阵文件初始化
        with open(self.dm_state_dat_path, "wb") as f:
            # Tag: 21 bytes + padding (3 bytes) = 24 bytes
            tag = b'XQI-QC-Density-Matrix'
            f.write(tag.ljust(24, b'\x00'))
            f.write(struct.pack('<I', self.env.qreg_size))
            f.write(struct.pack('<I', self.shot_total))

    def _append_shot_states_to_binary(self):
        """每轮 Shot 追加数据，确保使用小端序 float32"""
        # 写入状态矢量 (Before_measure_state)
        with open(self.state_dat_path, "ab") as f:
            for val in self.env.state_vector:
                # '<ff' 强制小端序，2个 float32 (real, imag)
                f.write(struct.pack('<ff', float(val.real), float(val.imag)))

        # 写入密度矩阵
        with open(self.dm_state_dat_path, "ab") as f:
            dim = self.env.density_matrix.shape[0]
            for i in range(dim):
                for j in range(dim):
                    val = self.env.density_matrix[i, j]
                    f.write(struct.pack('<ff', float(val.real), float(val.imag)))

    def _finalize_binary_files(self):
        """末尾写入对齐的 COUNT 统计信息"""
        # 写入状态矢量统计
        with open(self.state_dat_path, "ab") as f:
            # 'COUNT' (5 bytes) + padding (3 bytes) = 8 bytes
            f.write(b'COUNT'.ljust(8, b'\x00'))
            for count in self.shots_count_sv:
                f.write(struct.pack('<I', int(count)))

        # 写入密度矩阵统计
        with open(self.dm_state_dat_path, "ab") as f:
            f.write(b'COUNT'.ljust(8, b'\x00'))
            for count in self.shots_count_dm:
                f.write(struct.pack('<I', int(count)))

    def print_debug_info(self, shot_id=1):
        if not self._debug_file_initialized:
            self._initialize_debug_files()

        q_size = self.env.qreg_size
        dim = 2 ** q_size

        # 直接获取两种状态，不再进行模式转换判断
        psi = self.env.state_vector
        rho = self.env.density_matrix

        # --- 1. 写入 State Vector 文本 (XQI-QC-list.txt) ---
        with open(self.paths['sv_txt'], "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")
            f.write("XQI: current states (High Qubit->Low Qubit) :\n")
            for i in range(dim):
                bin_str = format(i, f'0{q_size}b')
                f.write(f"state  {bin_str:>10s}:  ({psi[i].real:.6f})+({psi[i].imag:.6f})i\n")

            f.write("corresponding Density Matrix state:\n")
            # 状态矢量对应的密度矩阵是其自身的外积
            self._write_matrix_to_text(f, np.outer(psi, psi.conj()))

            f.write("\ncurrent states probability (High Qubit->Low Qubit) :\n")
            for i in range(dim):
                bin_str = format(i, f'0{q_size}b')
                f.write(f"state {bin_str:>10s}:  {np.abs(psi[i]) ** 2:.6f}\n")
            self._write_common_debug(f)

        # --- 2. 写入 Density Matrix 文本 (XQI-QC-Density-Matrix-list.txt) ---
        with open(self.paths['dm_txt'], "a", encoding="utf-8") as f:
            f.write(f"debuging:  current status:  PC={self.env.pc}\n")
            f.write(f"shot: {shot_id}\n")
            f.write("XQI: current Density Matrix states:\n")
            self._write_matrix_to_text(f, rho)
            self._write_common_debug(f)

        # --- 3. 写入二进制数据 (.dat) ---
        # 显式使用 float32 确保与 C 语言 float 兼容;
        # 统一 '<ff' 显式小端,与 state .dat 的写入格式保持一致
        with open(self.paths['sv_dat'], "ab") as f:
            for val in psi:
                f.write(struct.pack('<ff', float(val.real), float(val.imag)))

        with open(self.paths['dm_dat'], "ab") as f:
            for i in range(dim):
                for j in range(dim):
                    val = rho[i, j]
                    f.write(struct.pack('<ff', float(val.real), float(val.imag)))
    def _initialize_debug_files(self):
        """
        初始化所有调试文件。
        如果是本次运行的第一次 debug，则删除旧文件并写入 Header。
        """
        log_dir = os.path.dirname(self.parser.source_path) if self.parser.source_path else os.getcwd()
        # 统一定义文件路径
        self.paths = {
            'sv_txt': os.path.join(log_dir, config.FILENAME_DEBUG),
            'sv_dat': os.path.join(log_dir, config.FILENAME_DEBUG_MATLAB),
            'dm_txt': os.path.join(log_dir, config.FILENAME_DEBUG_DENSITY_MATRIX),
            'dm_dat': os.path.join(log_dir, config.FILENAME_DEBUG_DENSITY_MATRIX_MATLAB)
        }
        # --- 第一步：物理删除已存在的旧文件 ---
        for path in self.paths.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    # print(f"Debug file cleaned: {os.path.basename(path)}")
                except OSError as e:
                    print(f"Error cleaning debug file {path}: {e}")
        # --- 第二步：初始化文本文件 (Header) ---
        header_text = self.source_code_text.strip() + "\n\n"
        header_text += "Label Number Sequence Label Symbol\n"
        for idx, (seq, symbol) in enumerate(self.labels_info):
            header_text += f"Label {idx:3d}: {seq:3d} {symbol}\n"
        header_text += "\n\n"
        with open(self.paths['sv_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)
        with open(self.paths['dm_txt'], "w", encoding="utf-8") as f:
            f.write(header_text)
        # --- 第三步：初始化二进制文件 (Header) ---
        # 1. SV Binary Header: Tag(6 bytes) + QregSize(4 bytes)
        with open(self.paths['sv_dat'], "wb") as f:
            f.write(b'XQI-QC') # 对应 C 语言 char[6]
            f.write(struct.pack('I', self.env.qreg_size)) # 对应 C 语言 unsigned int
        # 2. DM Binary Header: Tag(21 bytes) + QregSize(4 bytes)
        with open(self.paths['dm_dat'], "wb") as f:
            tag_dm = b'XQI-QC-Density-Matrix' # 21字节
            f.write(tag_dm)
            f.write(struct.pack('I', self.env.qreg_size))
        # 标记初始化完成
        self._debug_file_initialized = True
    @staticmethod
    def _write_matrix_to_text(f, matrix):
        rows, cols = matrix.shape
        f.write(f"\nmatrix rows:{rows}, matrix columns:{cols}:\n")
        for i in range(rows):
            for j in range(cols):
                val = matrix[i, j]
                f.write(f"[{i}][{j}]:({val.real:.6f})+({val.imag:.6f})i\n")
        f.write("\n")
    def _write_common_debug(self, f):
        """写入寄存器、CPSR 和内存的通用部分"""
        f.write("\n register:\n")
        for idx, val in enumerate(self.env.registers):
            f.write(f"R[{idx:2d}]={val:15.10f}\n")
        f.write("\nCPSR: ")
        f.write(f"SIGN_FLAG={self.env.SF}; " if self.env.SF else "SIGN_FLAG=0; ")
        f.write(f"ZERO_FLAG={self.env.ZF}.\n" if self.env.ZF else "ZERO_FLAG=0.\n")
        f.write("\n memory:\n")
        for idx, val in enumerate(self.env.memory):
            f.write(f"M[{idx:4d}]={val:15.10f}\n")
        f.write("\n\n")

    @staticmethod
    def _calculate_state_code(creg_array):
        """通用代码计算：将经典寄存器数组转为整数索引"""
        code = 0
        for i, val in enumerate(creg_array):
            bit = 1 if abs(val.real) > 0.5 else 0
            code += (bit << i)
        return code

    def _print_complete_measure_info(self, state_vec, filter_unmeasured=False):
        print("\nXQI Success: Complete Measure Event Information:\n")
        dim = len(state_vec)
        q_size = self.env.qreg_size
        measured_bits = getattr(self.env, 'measured_bits', set())

        for i in range(dim):
            bin_str = format(i, f'0{q_size}b')
            prob = np.abs(state_vec[i]) ** 2

            # C 语言逻辑：对于没被测量到的位，如果状态在该位上是 '1'，则该路径概率为 0
            if filter_unmeasured:
                for q_idx in range(q_size):
                    if q_idx not in measured_bits:
                        # 匹配 C 语言：Transform_Unsigned_Decimal_to_Binary 的索引
                        if bin_str[q_size - 1 - q_idx] == '1':
                            prob = 0.0
                            break

            print(f"state:        {bin_str}: probability={prob:.6f}")
            print("resultant measure state:")
            print(f"matrix rows:{dim}, matrix columns:1:")
            for j in range(dim):
                # 只有对应基矢的分量保留，其他为 0 (模拟投影)
                val = state_vec[j] if (i == j and prob > 0) else 0.0j
                print(f"[{j}][0]:({val.real:.6f})+({val.imag:.6f})i")
            print("")

    def _print_final_counts(self):
        q_size = self.env.qreg_size
        num_states = 2 ** q_size

        print("\nXQI Success: After measure, STATE COUNT:")
        print(f"Total Count:   {self.shot_total}")
        for i in range(num_states):
            bin_str = format(i, f'0{q_size}b')
            count = self.shots_count_sv[i]
            print(f"State:        {bin_str}:     Count={count:6d},    Probability={count / self.shot_total:.6f}")

        print("\n\nXQI Success: After measure, STATE COUNT (Density Matrix):")
        print(f"Total Count:   {self.shot_total}")
        for i in range(num_states):
            bin_str = format(i, f'0{q_size}b')
            count = self.shots_count_dm[i]
            print(f"State:        {bin_str}:     Count={count:6d},    Probability={count / self.shot_total:.6f}")
