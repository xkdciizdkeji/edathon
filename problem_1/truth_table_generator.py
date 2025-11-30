"""
三值逻辑真值表生成器
用于生成三值逻辑（值域为{0, 1, 2}）的随机或指定的真值表

三值逻辑基础：
- 输入和输出都在 {0, 1, 2} 范围内
- 一个 k 输入的门有 3^k 种输入组合，每种组合对应一个输出值
"""

import random
import itertools
from typing import Dict, Tuple, List, Optional


def generate_all_input_patterns(num_inputs: int) -> List[Tuple[int, ...]]:
    """
    生成所有可能的输入模式
    
    Args:
        num_inputs: 输入变量的数量
    
    Returns:
        所有 3^num_inputs 种输入模式的列表
    """
    return list(itertools.product([0, 1, 2], repeat=num_inputs))


def generate_random_truth_table(num_inputs: int, gate_name: str = "GATE") -> Dict:
    """
    生成一个随机的三值真值表
    
    Args:
        num_inputs: 输入变量的数量
        gate_name: 门的名称
    
    Returns:
        真值表字典，包含门名称、输入数量和映射
    """
    patterns = generate_all_input_patterns(num_inputs)
    # 为每种输入模式随机分配一个输出值 {0, 1, 2}
    mapping = {pattern: random.choice([0, 1, 2]) for pattern in patterns}
    
    return {
        "name": gate_name,
        "num_inputs": num_inputs,
        "mapping": mapping
    }


def generate_standard_gates() -> Dict[str, Dict]:
    """
    生成一些标准的三值逻辑门
    
    这里我们定义一些常用的三值逻辑操作：
    - MIN (三值 AND): min(a, b)
    - MAX (三值 OR): max(a, b)
    - NOT (三值取反): 2 - a
    - MOD3_ADD: (a + b) mod 3
    - MOD3_MUL: (a * b) mod 3
    
    Returns:
        标准门的字典
    """
    gates = {}
    
    # MIN gate (类似AND)
    min_mapping = {}
    for a, b in itertools.product([0, 1, 2], repeat=2):
        min_mapping[(a, b)] = min(a, b)
    gates["MIN"] = {
        "name": "MIN",
        "num_inputs": 2,
        "mapping": min_mapping
    }
    
    # MAX gate (类似OR)
    max_mapping = {}
    for a, b in itertools.product([0, 1, 2], repeat=2):
        max_mapping[(a, b)] = max(a, b)
    gates["MAX"] = {
        "name": "MAX",
        "num_inputs": 2,
        "mapping": max_mapping
    }
    
    # NOT gate (取反: 2 - a)
    not_mapping = {}
    for a in [0, 1, 2]:
        not_mapping[(a,)] = 2 - a
    gates["NOT"] = {
        "name": "NOT",
        "num_inputs": 1,
        "mapping": not_mapping
    }
    
    # MOD3_ADD gate
    add_mapping = {}
    for a, b in itertools.product([0, 1, 2], repeat=2):
        add_mapping[(a, b)] = (a + b) % 3
    gates["MOD3_ADD"] = {
        "name": "MOD3_ADD",
        "num_inputs": 2,
        "mapping": add_mapping
    }
    
    # MOD3_MUL gate
    mul_mapping = {}
    for a, b in itertools.product([0, 1, 2], repeat=2):
        mul_mapping[(a, b)] = (a * b) % 3
    gates["MOD3_MUL"] = {
        "name": "MOD3_MUL",
        "num_inputs": 2,
        "mapping": mul_mapping
    }
    
    # 三输入 MUX (多路选择器): 根据第一个输入选择后两个输入之一
    # mux(s, a, b) = a if s == 0, b if s == 2, else (a+b)//2 的某种组合
    mux_mapping = {}
    for s, a, b in itertools.product([0, 1, 2], repeat=3):
        if s == 0:
            mux_mapping[(s, a, b)] = a
        elif s == 2:
            mux_mapping[(s, a, b)] = b
        else:  # s == 1
            mux_mapping[(s, a, b)] = (a + b) % 3  # 某种混合
    gates["MUX3"] = {
        "name": "MUX3",
        "num_inputs": 3,
        "mapping": mux_mapping
    }
    
    return gates


def save_truth_table_to_file(truth_table: Dict, filename: str):
    """
    将真值表保存到文件
    
    格式：
    # GATE_NAME num_inputs
    input1,input2,...,inputk -> output
    
    Args:
        truth_table: 真值表字典
        filename: 输出文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {truth_table['name']} {truth_table['num_inputs']}\n")
        for pattern, output in sorted(truth_table['mapping'].items()):
            input_str = ",".join(map(str, pattern))
            f.write(f"{input_str} -> {output}\n")


def load_truth_table_from_file(filename: str) -> Dict:
    """
    从文件加载真值表
    
    Args:
        filename: 输入文件名
    
    Returns:
        真值表字典
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析第一行：# GATE_NAME num_inputs
    header = lines[0].strip()
    parts = header.split()
    gate_name = parts[1]
    num_inputs = int(parts[2])
    
    # 解析映射
    mapping = {}
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        input_part, output_part = line.split('->')
        inputs = tuple(map(int, input_part.strip().split(',')))
        output = int(output_part.strip())
        mapping[inputs] = output
    
    return {
        "name": gate_name,
        "num_inputs": num_inputs,
        "mapping": mapping
    }


def print_truth_table(truth_table: Dict):
    """打印真值表"""
    print(f"Gate: {truth_table['name']}, Inputs: {truth_table['num_inputs']}")
    print("-" * 30)
    for pattern, output in sorted(truth_table['mapping'].items()):
        input_str = ", ".join(map(str, pattern))
        print(f"({input_str}) -> {output}")


if __name__ == "__main__":
    # 演示：生成并打印标准门
    print("=== 标准三值逻辑门 ===\n")
    standard_gates = generate_standard_gates()
    
    for name, gate in standard_gates.items():
        print_truth_table(gate)
        print()
    
    # 生成一个随机的2输入门
    print("=== 随机生成的2输入门 ===\n")
    random_gate = generate_random_truth_table(2, "RANDOM_GATE")
    print_truth_table(random_gate)
    
    # 保存到文件
    save_truth_table_to_file(standard_gates["MIN"], "test_min_gate.txt")
    print("\n已将MIN门保存到 test_min_gate.txt")
