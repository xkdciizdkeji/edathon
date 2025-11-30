"""
测试电路生成器
用于生成三值逻辑电路（BENCH格式）以测试重替换算法

关键思想：
为了方便验证算法的正确性，我们可以构造两类电路：
1. 正例：最后一个输出确实可以由其他输出表示（有替换解）
2. 反例：最后一个输出不能由其他输出表示（无替换解）

对于正例，我们使用你提出的idea：
- 先定义 n1, n2, ..., n_{k-1} 作为前k-1个输出（它们是主输入的某些函数）
- 然后定义 n_k = f(n1, n2, ..., n_{k-1})（n_k是其他输出的函数）
- 接着把这个表达式完全展开成主输入的函数形式（支持嵌套门）
- 这样我们就知道正确答案是 f

对于反例：
- 构造一个输出，使其在某些输入组合下与其他输出的组合有冲突

注意：BENCH格式支持嵌套表达式，如 o2 = MIN(MIN(i0, i1), MAX(i0, i2))
"""

import random
import itertools
from typing import Dict, Tuple, List, Optional, Set
from truth_table_generator import generate_standard_gates, generate_all_input_patterns


class Circuit:
    """
    三值逻辑电路类
    
    支持两种门定义方式：
    1. 简单形式：o0 = MIN(i0, i1)
    2. 嵌套形式：o2 = MIN(MIN(i0, i1), MAX(i0, i2))
    
    内部使用表达式树来表示嵌套结构
    """
    
    def __init__(self):
        self.primary_inputs: List[str] = []  # 主输入名称列表
        self.primary_outputs: List[str] = []  # 主输出名称列表
        # 门定义: {节点名: {gate_type, inputs}}
        # inputs可以是主输入名称，也可以是嵌套的门表达式（元组形式）
        self.gates: Dict[str, Dict] = {}
        self.gate_types: Dict[str, Dict] = {}  # 门类型的真值表
    
    def add_primary_input(self, name: str):
        """添加主输入"""
        self.primary_inputs.append(name)
    
    def add_primary_output(self, name: str):
        """添加主输出"""
        self.primary_outputs.append(name)
    
    def add_gate(self, output_name: str, gate_type: str, inputs: List):
        """
        添加一个门
        
        Args:
            output_name: 输出节点名称
            gate_type: 门类型（如 MIN, MAX, NOT）
            inputs: 输入列表，元素可以是：
                    - 字符串：主输入名称或其他门的输出名称
                    - 元组：嵌套的门表达式 (gate_type, [inputs...])
        """
        self.gates[output_name] = {
            "gate_type": gate_type,
            "inputs": inputs
        }
    
    def add_gate_type(self, gate_type: str, truth_table: Dict):
        """添加门类型定义"""
        self.gate_types[gate_type] = truth_table
    
    def _evaluate_expression(self, expr, values: Dict[str, int]) -> int:
        """
        递归计算表达式的值
        
        Args:
            expr: 表达式，可以是：
                  - 字符串：变量名
                  - 元组：(gate_type, [inputs...])
            values: 已知变量的值
        
        Returns:
            计算结果
        """
        if isinstance(expr, str):
            # 是变量名
            return values[expr]
        elif isinstance(expr, tuple):
            # 是嵌套表达式 (gate_type, inputs)
            gate_type, inputs = expr
            input_values = tuple(self._evaluate_expression(inp, values) for inp in inputs)
            return self.gate_types[gate_type]["mapping"][input_values]
        else:
            raise ValueError(f"未知的表达式类型: {type(expr)}")
    
    def simulate(self, input_values: Dict[str, int]) -> Dict[str, int]:
        """
        仿真电路，给定主输入值，计算所有节点的值
        
        Args:
            input_values: 主输入的值 {input_name: value}
        
        Returns:
            所有节点的值 {node_name: value}
        """
        values = dict(input_values)
        
        # 对于支持嵌套表达式的电路，我们需要递归计算
        for gate_name, gate in self.gates.items():
            gate_type = gate["gate_type"]
            inputs = gate["inputs"]
            
            # 计算每个输入的值（可能是嵌套表达式）
            input_values_list = []
            for inp in inputs:
                if isinstance(inp, str):
                    if inp in values:
                        input_values_list.append(values[inp])
                    elif inp in self.gates:
                        # 需要先计算这个门
                        # 简单情况下假设是拓扑有序的
                        raise ValueError(f"门 {inp} 尚未计算")
                    else:
                        raise ValueError(f"未知的输入: {inp}")
                elif isinstance(inp, tuple):
                    # 嵌套表达式，递归计算
                    input_values_list.append(self._evaluate_expression(inp, values))
                else:
                    raise ValueError(f"未知的输入类型: {type(inp)}")
            
            input_tuple = tuple(input_values_list)
            values[gate_name] = self.gate_types[gate_type]["mapping"][input_tuple]
        
        return values
    
    def get_output_values(self, input_values: Dict[str, int]) -> List[int]:
        """获取所有主输出的值（按顺序）"""
        all_values = self.simulate(input_values)
        return [all_values[out] for out in self.primary_outputs]
    
    def _expr_to_string(self, expr) -> str:
        """将表达式转换为字符串形式"""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, tuple):
            gate_type, inputs = expr
            inputs_str = ", ".join(self._expr_to_string(inp) for inp in inputs)
            return f"{gate_type}({inputs_str})"
        else:
            raise ValueError(f"未知的表达式类型: {type(expr)}")
    
    def to_bench_format(self) -> str:
        """将电路转换为BENCH格式字符串（支持嵌套表达式）"""
        lines = []
        lines.append("# Three-valued logic circuit (BENCH format)")
        lines.append("")
        
        # 主输入
        for pi in self.primary_inputs:
            lines.append(f"INPUT({pi})")
        lines.append("")
        
        # 主输出
        for po in self.primary_outputs:
            lines.append(f"OUTPUT({po})")
        lines.append("")
        
        # 门（支持嵌套表达式）
        for gate_name, gate in self.gates.items():
            gate_type = gate["gate_type"]
            inputs = gate["inputs"]
            # 将输入转换为字符串（支持嵌套）
            inputs_str = ", ".join(self._expr_to_string(inp) for inp in inputs)
            lines.append(f"{gate_name} = {gate_type}({inputs_str})")
        
        return "\n".join(lines)
        
        return "\n".join(lines)
    
    def save_to_bench(self, filename: str):
        """保存为BENCH文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.to_bench_format())
    
    def save_gate_types(self, filename: str):
        """保存门类型定义（真值表）"""
        with open(filename, 'w', encoding='utf-8') as f:
            for gate_type, tt in self.gate_types.items():
                f.write(f"# {gate_type} {tt['num_inputs']}\n")
                for pattern, output in sorted(tt['mapping'].items()):
                    input_str = ",".join(map(str, pattern))
                    f.write(f"{input_str} -> {output}\n")
                f.write("\n")


def generate_positive_case(
    num_primary_inputs: int = 3,
    num_other_outputs: int = 2,
    seed: Optional[int] = None
) -> Tuple[Circuit, str]:
    """
    生成一个正例：最后一个输出可以由其他输出表示
    
    关键：所有输出都必须直接由主输入表示（使用嵌套表达式，不引入中间变量）！
    
    策略：
    1. 创建 num_other_outputs 个输出 o0, o1, ... (每个是主输入的某个函数)
    2. 设计 o_last = f(o0, o1, ...)（这是我们期望的替换函数）
    3. 将 o_last 完全展开成主输入的嵌套函数形式
    4. 例如：o_last = MIN(MIN(i0, i1), MAX(i0, i2))
    
    Args:
        num_primary_inputs: 主输入数量
        num_other_outputs: 其他输出数量
        seed: 随机种子
    
    Returns:
        (电路, 预期的替换函数描述)
    """
    if seed is not None:
        random.seed(seed)
    
    circuit = Circuit()
    standard_gates = generate_standard_gates()
    
    # 添加标准门类型
    for name, gate in standard_gates.items():
        circuit.add_gate_type(name, gate)
    
    # 添加主输入
    input_names = [f"i{k}" for k in range(num_primary_inputs)]
    for name in input_names:
        circuit.add_primary_input(name)
    
    # 创建其他输出（每个是主输入的某个简单函数）
    # 记录每个输出的表达式结构，用于后续展开
    output_names = []
    output_expressions = {}  # {output_name: (gate_type, [inputs])} 作为嵌套表达式
    available_gates_2input = ["MIN", "MAX", "MOD3_ADD", "MOD3_MUL"]
    
    for k in range(num_other_outputs):
        output_name = f"o{k}"
        output_names.append(output_name)
        
        # 随机选择一个2输入门和两个主输入
        gate_type = random.choice(available_gates_2input)
        inp1, inp2 = random.sample(input_names, 2)
        circuit.add_gate(output_name, gate_type, [inp1, inp2])
        # 保存表达式结构（元组形式，用于嵌套）
        output_expressions[output_name] = (gate_type, [inp1, inp2])
    
    # 设计最后一个输出：o_last = f(o0, o1, ...)
    # 然后将其完全展开成主输入的嵌套表达式
    last_output_name = f"o{num_other_outputs}"
    
    if num_other_outputs >= 2:
        # 选择替换函数 f: o_last = resub_gate(o_i, o_j)
        resub_gate_type = random.choice(available_gates_2input)
        o1, o2 = random.sample(output_names, 2)
        resub_function = f"{last_output_name} = {resub_gate_type}({o1}, {o2})"
        
        # 获取 o1, o2 的表达式
        o1_expr = output_expressions[o1]  # (gate_type, [inputs])
        o2_expr = output_expressions[o2]  # (gate_type, [inputs])
        
        # o_last 使用嵌套表达式：resub_gate_type(o1_expr, o2_expr)
        # 输入是嵌套的元组表达式
        circuit.add_gate(last_output_name, resub_gate_type, [o1_expr, o2_expr])
        
    else:
        # 只有一个其他输出，用NOT
        resub_function = f"{last_output_name} = NOT({output_names[0]})"
        
        o1 = output_names[0]
        o1_expr = output_expressions[o1]
        
        # o_last = NOT(o1_expr)
        circuit.add_gate(last_output_name, "NOT", [o1_expr])
    
    # 添加所有输出（其他输出 + 最后输出）
    for name in output_names:
        circuit.add_primary_output(name)
    circuit.add_primary_output(last_output_name)
    
    return circuit, resub_function


def generate_negative_case(
    num_primary_inputs: int = 3,
    num_other_outputs: int = 2,
    seed: Optional[int] = None
) -> Circuit:
    """
    生成一个反例：最后一个输出不能由其他输出表示
    
    策略：
    1. 创建一些独立的输出（只依赖部分主输入）
    2. 创建一个最后输出，使得存在两组不同的主输入，
       它们在其他输出上的值相同，但最后输出值不同
    
    关键：使用嵌套表达式，不引入中间变量
    
    Args:
        num_primary_inputs: 主输入数量
        num_other_outputs: 其他输出数量
        seed: 随机种子
    
    Returns:
        电路（无替换解）
    """
    if seed is not None:
        random.seed(seed)
    
    circuit = Circuit()
    standard_gates = generate_standard_gates()
    
    # 添加标准门类型
    for name, gate in standard_gates.items():
        circuit.add_gate_type(name, gate)
    
    # 添加主输入
    input_names = [f"i{k}" for k in range(num_primary_inputs)]
    for name in input_names:
        circuit.add_primary_input(name)
    
    # 创建其他输出（每个只依赖部分主输入，留下一些"自由度"给最后输出造成冲突）
    output_names = []
    
    # 让所有其他输出只依赖前几个输入，最后输出依赖最后一个输入
    # 这样当前几个输入相同时，其他输出相同，但最后输出可能不同
    num_shared = min(num_primary_inputs - 1, max(1, num_primary_inputs - 1))
    shared_inputs = input_names[:num_shared]
    unique_input = input_names[-1]  # 最后一个输入只影响最后输出
    
    for k in range(num_other_outputs):
        output_name = f"o{k}"
        output_names.append(output_name)
        
        # 只使用共享输入
        if len(shared_inputs) >= 2:
            inp1, inp2 = random.sample(shared_inputs, 2)
            gate_type = random.choice(["MIN", "MAX", "MOD3_ADD"])
            circuit.add_gate(output_name, gate_type, [inp1, inp2])
        else:
            # 用NOT
            circuit.add_gate(output_name, "NOT", [shared_inputs[0]])
    
    # 最后输出同时依赖共享输入和独立输入（使用嵌套表达式）
    # 这确保了：相同的其他输出值 -> 不同的最后输出值（因为独立输入可变）
    last_output_name = f"o{num_other_outputs}"
    
    # 使用嵌套表达式：o_last = MOD3_ADD(MIN(shared[0], shared[1]), unique_input)
    if len(shared_inputs) >= 2:
        # 嵌套表达式：MOD3_ADD(MIN(i0, i1), i2)
        inner_expr = ("MIN", [shared_inputs[0], shared_inputs[1]])
        circuit.add_gate(last_output_name, "MOD3_ADD", [inner_expr, unique_input])
    else:
        # 简单情况：MOD3_ADD(i0, i_unique)
        circuit.add_gate(last_output_name, "MOD3_ADD", [shared_inputs[0], unique_input])
    
    # 添加所有输出
    for name in output_names:
        circuit.add_primary_output(name)
    circuit.add_primary_output(last_output_name)
    
    return circuit


def generate_complex_positive_case(
    num_primary_inputs: int = 4,
    num_other_outputs: int = 3,
    seed: Optional[int] = None
) -> Tuple[Circuit, str]:
    """
    生成一个更复杂的正例：所有输出都由主输入直接表示（使用嵌套表达式）
    
    关键：所有输出都必须直接由主输入表示，使用嵌套表达式！
    例如：o3 = MIN(MIN(i0, i1), MAX(i1, i2))
    
    Args:
        num_primary_inputs: 主输入数量
        num_other_outputs: 其他输出数量
        seed: 随机种子
    
    Returns:
        (电路, 预期的替换函数描述)
    """
    if seed is not None:
        random.seed(seed)
    
    circuit = Circuit()
    standard_gates = generate_standard_gates()
    
    for name, gate in standard_gates.items():
        circuit.add_gate_type(name, gate)
    
    # 主输入
    input_names = [f"i{k}" for k in range(num_primary_inputs)]
    for name in input_names:
        circuit.add_primary_input(name)
    
    available_gates_2input = ["MIN", "MAX", "MOD3_ADD", "MOD3_MUL"]
    
    # 创建其他输出（每个是主输入的函数）
    output_names = []
    output_expressions = {}  # 记录每个输出的表达式结构
    
    for k in range(num_other_outputs):
        output_name = f"o{k}"
        output_names.append(output_name)
        
        # 使用不同的输入组合，让输出有一定的多样性
        if num_primary_inputs >= 2:
            inp1_idx = k % num_primary_inputs
            inp2_idx = (k + 1) % num_primary_inputs
            inp1, inp2 = input_names[inp1_idx], input_names[inp2_idx]
            gate_type = available_gates_2input[k % len(available_gates_2input)]
            circuit.add_gate(output_name, gate_type, [inp1, inp2])
            output_expressions[output_name] = (gate_type, [inp1, inp2])
        else:
            circuit.add_gate(output_name, "NOT", [input_names[0]])
            output_expressions[output_name] = ("NOT", [input_names[0]])
    
    # 设计最后一个输出：o_last = f(o_i, o_j)
    # 然后将其完全展开成主输入的嵌套表达式
    last_output_name = f"o{num_other_outputs}"
    
    if num_other_outputs >= 2:
        # 选择替换函数
        resub_gate_type = random.choice(available_gates_2input)
        o1, o2 = output_names[0], output_names[1]
        resub_function = f"{last_output_name} = {resub_gate_type}({o1}, {o2})"
        
        # 获取 o1, o2 的表达式
        o1_expr = output_expressions[o1]
        o2_expr = output_expressions[o2]
        
        # o_last 使用嵌套表达式
        circuit.add_gate(last_output_name, resub_gate_type, [o1_expr, o2_expr])
    else:
        # 只有一个其他输出
        resub_function = f"{last_output_name} = NOT({output_names[0]})"
        o1_expr = output_expressions[output_names[0]]
        circuit.add_gate(last_output_name, "NOT", [o1_expr])
    
    # 添加输出
    for name in output_names:
        circuit.add_primary_output(name)
    circuit.add_primary_output(last_output_name)
    
    return circuit, resub_function


def generate_large_scale_circuit(
    num_primary_inputs: int = 50,
    num_other_outputs: int = 10,
    num_intermediate_layers: int = 3,
    gates_per_layer: int = 20,
    is_positive: bool = True,
    seed: Optional[int] = None
) -> Tuple[Circuit, Optional[str]]:
    """
    生成大规模复杂电路
    
    【设计思想】
    
    为了测试 S&S 算法在大规模电路上的性能，我们需要生成：
    1. 大量主输入（如 50 个）
    2. 多层中间逻辑（增加电路深度和复杂度）
    3. 多个输出（每个输出是中间逻辑的函数）
    
    【电路结构】
    
    主输入层: i0, i1, ..., i_{n-1}
        ↓
    中间层1: 随机组合主输入，生成 gates_per_layer 个中间节点
        ↓
    中间层2: 随机组合上一层和主输入
        ↓
    ...
        ↓
    输出层: 每个输出是中间节点的函数
    
    【正例 vs 反例】
    
    正例：最后输出 = f(其他输出)，展开成主输入的函数
    反例：最后输出依赖一个"独立"的主输入（其他输出不依赖它）
    
    Args:
        num_primary_inputs: 主输入数量
        num_other_outputs: 其他输出数量
        num_intermediate_layers: 中间层数量
        gates_per_layer: 每层的门数量
        is_positive: True生成正例，False生成反例
        seed: 随机种子
    
    Returns:
        (电路, 预期替换函数描述 或 None)
    """
    if seed is not None:
        random.seed(seed)
    
    circuit = Circuit()
    standard_gates = generate_standard_gates()
    
    # 添加标准门类型
    for name, gate in standard_gates.items():
        circuit.add_gate_type(name, gate)
    
    # 添加主输入
    input_names = [f"i{k}" for k in range(num_primary_inputs)]
    for name in input_names:
        circuit.add_primary_input(name)
    
    available_gates_2input = ["MIN", "MAX", "MOD3_ADD", "MOD3_MUL"]
    
    # 构建中间层表达式（不作为输出，只作为构建块）
    # 每一层的节点可以引用：主输入 + 之前所有层的节点
    all_expressions = list(input_names)  # 初始可用的表达式就是主输入
    layer_expressions = [list(input_names)]  # 每层的表达式列表
    
    for layer_idx in range(num_intermediate_layers):
        new_layer = []
        
        for gate_idx in range(gates_per_layer):
            # 随机选择门类型
            gate_type = random.choice(available_gates_2input)
            
            # 随机选择两个输入（从所有可用表达式中）
            # 倾向于选择最近的层，以增加深度
            if layer_idx > 0 and random.random() < 0.7:
                # 70% 概率从最近的层选择至少一个输入
                inp1 = random.choice(layer_expressions[-1])
                inp2 = random.choice(all_expressions)
            else:
                inp1, inp2 = random.sample(all_expressions, 2)
            
            # 创建嵌套表达式
            expr = (gate_type, [inp1, inp2])
            new_layer.append(expr)
        
        layer_expressions.append(new_layer)
        all_expressions.extend(new_layer)
    
    # 从最后一层选择表达式来构建其他输出
    final_layer = layer_expressions[-1] if layer_expressions[-1] else all_expressions
    
    output_names = []
    output_expressions = {}
    
    for k in range(num_other_outputs):
        output_name = f"o{k}"
        output_names.append(output_name)
        
        # 选择一个表达式作为输出
        if k < len(final_layer):
            expr = final_layer[k]
        else:
            # 如果输出数量超过最后一层的节点数，随机组合
            gate_type = random.choice(available_gates_2input)
            inp1 = random.choice(final_layer) if final_layer else random.choice(input_names)
            inp2 = random.choice(all_expressions)
            expr = (gate_type, [inp1, inp2])
        
        # 添加到电路
        if isinstance(expr, str):
            # 如果是简单的主输入引用，包装成门
            circuit.add_gate(output_name, "MIN", [expr, expr])  # x = MIN(x, x) = x
            output_expressions[output_name] = ("MIN", [expr, expr])
        elif isinstance(expr, tuple):
            gate_type, inputs = expr
            circuit.add_gate(output_name, gate_type, inputs)
            output_expressions[output_name] = expr
    
    # 构建最后一个输出
    last_output_name = f"o{num_other_outputs}"
    resub_function = None
    
    if is_positive:
        # 正例：最后输出是其他输出的函数
        # 选择两个其他输出，用一个门组合它们
        if num_other_outputs >= 2:
            resub_gate_type = random.choice(available_gates_2input)
            o1, o2 = random.sample(output_names, 2)
            resub_function = f"{last_output_name} = {resub_gate_type}({o1}, {o2})"
            
            # 获取 o1, o2 的表达式并展开
            o1_expr = output_expressions[o1]
            o2_expr = output_expressions[o2]
            
            # 最后输出使用嵌套表达式
            circuit.add_gate(last_output_name, resub_gate_type, [o1_expr, o2_expr])
        else:
            # 只有一个其他输出，用 NOT
            resub_function = f"{last_output_name} = NOT({output_names[0]})"
            o1_expr = output_expressions[output_names[0]]
            circuit.add_gate(last_output_name, "NOT", [o1_expr])
    else:
        # 反例：最后输出依赖一个"独立"的主输入
        # 选择一个其他输出都不依赖的主输入
        
        # 找出其他输出依赖的所有主输入
        def get_input_deps(expr) -> Set[str]:
            """递归获取表达式依赖的主输入"""
            if isinstance(expr, str):
                if expr in input_names:
                    return {expr}
                return set()
            elif isinstance(expr, tuple):
                _, inputs = expr
                deps = set()
                for inp in inputs:
                    deps |= get_input_deps(inp)
                return deps
            return set()
        
        used_inputs = set()
        for out_name in output_names:
            used_inputs |= get_input_deps(output_expressions[out_name])
        
        # 找一个未被使用的主输入
        unused_inputs = set(input_names) - used_inputs
        
        if unused_inputs:
            independent_input = random.choice(list(unused_inputs))
        else:
            # 如果所有输入都被使用了，我们需要用不同的策略
            # 选择一个输入，让最后输出直接依赖它，而其他输出通过复杂路径依赖它
            independent_input = input_names[-1]  # 使用最后一个输入
        
        # 最后输出 = 某个其他输出的表达式 + 独立输入
        if output_names:
            base_expr = output_expressions[output_names[0]]
            circuit.add_gate(last_output_name, "MOD3_ADD", [base_expr, independent_input])
        else:
            circuit.add_gate(last_output_name, "MOD3_ADD", [input_names[0], independent_input])
    
    # 添加所有输出
    for name in output_names:
        circuit.add_primary_output(name)
    circuit.add_primary_output(last_output_name)
    
    return circuit, resub_function


def generate_scalable_test_suite(
    base_dir: str = "test_cases",
    scales: List[Tuple[int, int, int]] = None
) -> List[Tuple[str, Circuit, bool]]:
    """
    生成一系列不同规模的测试用例
    
    Args:
        base_dir: 保存目录
        scales: [(num_inputs, num_outputs, num_layers), ...] 规模列表
    
    Returns:
        [(文件名, 电路, 是否为正例), ...]
    """
    import os
    os.makedirs(base_dir, exist_ok=True)
    
    if scales is None:
        scales = [
            # (输入数, 输出数, 中间层数)
            (10, 3, 2),    # 小规模
            (20, 5, 3),    # 中等规模
            (30, 8, 3),    # 较大规模
            (50, 10, 4),   # 大规模
            (100, 15, 5),  # 超大规模
        ]
    
    results = []
    
    for num_inputs, num_outputs, num_layers in scales:
        for is_positive in [True, False]:
            case_type = "positive" if is_positive else "negative"
            case_name = f"scale_{num_inputs}in_{num_outputs}out_{case_type}"
            
            print(f"生成 {case_name}...")
            
            circuit, resub_func = generate_large_scale_circuit(
                num_primary_inputs=num_inputs,
                num_other_outputs=num_outputs - 1,
                num_intermediate_layers=num_layers,
                gates_per_layer=max(10, num_inputs // 3),
                is_positive=is_positive,
                seed=hash(case_name) % (2**31)
            )
            
            # 保存
            base_path = os.path.join(base_dir, case_name)
            circuit.save_to_bench(f"{base_path}.bench")
            circuit.save_gate_types(f"{base_path}_gates.txt")
            
            # 保存信息
            with open(f"{base_path}_info.txt", 'w', encoding='utf-8') as f:
                f.write(f"测试用例: {case_name}\n")
                f.write(f"主输入数: {len(circuit.primary_inputs)}\n")
                f.write(f"主输出数: {len(circuit.primary_outputs)}\n")
                f.write(f"预期结果: {'可替换' if is_positive else '不可替换'}\n")
                if resub_func:
                    f.write(f"替换函数: {resub_func}\n")
                f.write(f"输入空间大小: 3^{len(circuit.primary_inputs)} = {3**len(circuit.primary_inputs)}\n")
            
            results.append((case_name, circuit, is_positive))
            print(f"  输入: {len(circuit.primary_inputs)}, 输出: {len(circuit.primary_outputs)}")
            print(f"  输入空间: 3^{len(circuit.primary_inputs)} ≈ {3**len(circuit.primary_inputs):.2e}")
    
    return results


def verify_resubstitution_exists(circuit: Circuit) -> Tuple[bool, Optional[str]]:
    """
    验证电路的最后一个输出是否可由其他输出表示
    
    通过枚举所有输入组合，检查是否存在冲突
    
    Args:
        circuit: 待验证的电路
    
    Returns:
        (是否可替换, 冲突描述或None)
    """
    num_inputs = len(circuit.primary_inputs)
    all_patterns = generate_all_input_patterns(num_inputs)
    
    # 收集 (其他输出组合) -> 最后输出值 的映射
    output_map: Dict[Tuple[int, ...], Set[int]] = {}
    
    for pattern in all_patterns:
        input_values = {name: val for name, val in zip(circuit.primary_inputs, pattern)}
        outputs = circuit.get_output_values(input_values)
        
        other_outputs = tuple(outputs[:-1])
        last_output = outputs[-1]
        
        if other_outputs not in output_map:
            output_map[other_outputs] = set()
        output_map[other_outputs].add(last_output)
        
        # 如果同一个其他输出组合对应多个不同的最后输出值，则不可替换
        if len(output_map[other_outputs]) > 1:
            return False, f"冲突：其他输出={other_outputs}, 最后输出可为{output_map[other_outputs]}"
    
    return True, None


def save_test_case(circuit: Circuit, base_filename: str, description: str = ""):
    """保存测试用例"""
    circuit.save_to_bench(f"{base_filename}.bench")
    circuit.save_gate_types(f"{base_filename}_gates.txt")
    
    # 保存描述
    with open(f"{base_filename}_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"测试用例: {base_filename}\n")
        f.write(f"描述: {description}\n")
        f.write(f"主输入: {circuit.primary_inputs}\n")
        f.write(f"主输出: {circuit.primary_outputs}\n")
        f.write(f"最后输出: {circuit.primary_outputs[-1]}\n")
        f.write(f"其他输出: {circuit.primary_outputs[:-1]}\n")


if __name__ == "__main__":
    import os
    import sys
    
    # 创建测试用例目录
    test_dir = "test_cases"
    os.makedirs(test_dir, exist_ok=True)
    
    # 检查是否有命令行参数
    generate_large = "--large" in sys.argv or "-l" in sys.argv
    
    print("=== 生成测试用例 ===\n")
    
    # 生成正例1：简单的2输出电路
    print("生成正例1（简单）...")
    circuit1, resub1 = generate_positive_case(num_primary_inputs=3, num_other_outputs=2, seed=42)
    can_resub, conflict = verify_resubstitution_exists(circuit1)
    print(f"  可替换: {can_resub}")
    print(f"  预期替换函数: {resub1}")
    save_test_case(circuit1, f"{test_dir}/positive_simple", f"正例-简单，{resub1}")
    print(f"  已保存到 {test_dir}/positive_simple.*")
    print()
    
    # 生成正例2：复杂的多层电路
    print("生成正例2（复杂）...")
    circuit2, resub2 = generate_complex_positive_case(num_primary_inputs=4, num_other_outputs=3, seed=123)
    can_resub, conflict = verify_resubstitution_exists(circuit2)
    print(f"  可替换: {can_resub}")
    print(f"  预期替换函数: {resub2}")
    save_test_case(circuit2, f"{test_dir}/positive_complex", f"正例-复杂，{resub2}")
    print(f"  已保存到 {test_dir}/positive_complex.*")
    print()
    
    # 生成反例：最后输出不能由其他输出表示
    print("生成反例...")
    circuit3 = generate_negative_case(num_primary_inputs=3, num_other_outputs=2, seed=456)
    can_resub, conflict = verify_resubstitution_exists(circuit3)
    print(f"  可替换: {can_resub}")
    if conflict:
        print(f"  冲突: {conflict}")
    save_test_case(circuit3, f"{test_dir}/negative_case", "反例-不可替换")
    print(f"  已保存到 {test_dir}/negative_case.*")
    print()
    
    if generate_large:
        print("=" * 60)
        print("=== 生成大规模测试用例（--large 模式）===")
        print("=" * 60)
        print()
        
        # 定义不同规模
        large_scales = [
            # (输入数, 输出数, 中间层数)
            (10, 4, 2),     # 小规模：3^10 ≈ 59,049
            (15, 5, 2),     # 中等规模：3^15 ≈ 14,348,907
            (20, 6, 3),     # 较大规模：3^20 ≈ 3.49×10^9
            (30, 8, 3),     # 大规模：3^30 ≈ 2.06×10^14
            (50, 10, 4),    # 超大规模：3^50 ≈ 7.18×10^23
        ]
        
        for num_inputs, num_outputs, num_layers in large_scales:
            for is_positive in [True, False]:
                case_type = "positive" if is_positive else "negative"
                case_name = f"large_{num_inputs}in_{num_outputs}out_{case_type}"
                
                print(f"\n生成 {case_name}...")
                
                circuit, resub_func = generate_large_scale_circuit(
                    num_primary_inputs=num_inputs,
                    num_other_outputs=num_outputs - 1,
                    num_intermediate_layers=num_layers,
                    gates_per_layer=max(10, num_inputs // 2),
                    is_positive=is_positive,
                    seed=hash(case_name) % (2**31)
                )
                
                # 保存
                base_path = f"{test_dir}/{case_name}"
                circuit.save_to_bench(f"{base_path}.bench")
                circuit.save_gate_types(f"{base_path}_gates.txt")
                
                # 保存信息
                with open(f"{base_path}_info.txt", 'w', encoding='utf-8') as f:
                    f.write(f"测试用例: {case_name}\n")
                    f.write(f"主输入数: {len(circuit.primary_inputs)}\n")
                    f.write(f"主输出数: {len(circuit.primary_outputs)}\n")
                    f.write(f"预期结果: {'可替换' if is_positive else '不可替换'}\n")
                    if resub_func:
                        f.write(f"替换函数: {resub_func}\n")
                    f.write(f"输入空间大小: 3^{len(circuit.primary_inputs)} ≈ {3**len(circuit.primary_inputs):.2e}\n")
                
                print(f"  输入: {len(circuit.primary_inputs)}, 输出: {len(circuit.primary_outputs)}")
                print(f"  输入空间: 3^{len(circuit.primary_inputs)} ≈ {3**len(circuit.primary_inputs):.2e}")
                print(f"  预期结果: {'可替换' if is_positive else '不可替换'}")
                if resub_func:
                    print(f"  替换函数: {resub_func}")
                print(f"  已保存到 {base_path}.*")
        
        print("\n" + "=" * 60)
        print("大规模测试用例生成完成！")
        print("=" * 60)
    else:
        print("-" * 50)
        print("提示: 使用 --large 或 -l 参数生成大规模测试用例")
        print("例如: python circuit_generator.py --large")
        print("-" * 50)
    
    print()
    
    # 打印一个电路的BENCH格式示例
    print("=== 正例1的BENCH格式 ===")
    print(circuit1.to_bench_format())
