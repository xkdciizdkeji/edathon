"""
BENCH格式解析器
用于解析三值逻辑电路的BENCH文件和真值表文件

BENCH格式示例：
INPUT(a)
INPUT(b)
OUTPUT(n1)
OUTPUT(n2)
n1 = AND(a, b)
n2 = OR(a, n1)

真值表文件格式：
# GATE_NAME num_inputs
0,0 -> 0
0,1 -> 0
...
"""

import re
from typing import Dict, List, Tuple, Optional


class BenchParser:
    """BENCH格式电路解析器"""
    
    def __init__(self):
        self.primary_inputs: List[str] = []
        self.primary_outputs: List[str] = []
        self.gates: Dict[str, Dict] = {}  # {节点名: {gate_type, inputs}}
        self.gate_types: Dict[str, Dict] = {}  # {门类型名: {num_inputs, mapping}}
    
    def parse_bench_file(self, filename: str):
        """
        解析BENCH文件
        
        Args:
            filename: BENCH文件路径
        """
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        self.parse_bench_string(content)
    
    def _parse_expression(self, expr_str: str):
        """
        解析表达式字符串，支持嵌套
        
        例如：
        - "i0" -> "i0"
        - "MIN(i0, i1)" -> ("MIN", ["i0", "i1"])
        - "MIN(MIN(i0, i1), MAX(i0, i2))" -> ("MIN", [("MIN", ["i0", "i1"]), ("MAX", ["i0", "i2"])])
        
        Args:
            expr_str: 表达式字符串
        
        Returns:
            解析后的表达式（字符串或元组）
        """
        expr_str = expr_str.strip()
        
        # 检查是否是简单变量名
        if re.match(r'^\w+$', expr_str):
            return expr_str
        
        # 解析 GATE(args) 形式
        match = re.match(r'^(\w+)\s*\((.*)\)$', expr_str, re.DOTALL)
        if match:
            gate_type = match.group(1)
            args_str = match.group(2)
            
            # 解析参数（需要处理嵌套括号）
            args = self._split_args(args_str)
            parsed_args = [self._parse_expression(arg) for arg in args]
            
            return (gate_type, parsed_args)
        
        raise ValueError(f"无法解析表达式: {expr_str}")
    
    def _split_args(self, args_str: str) -> List[str]:
        """
        分割函数参数，正确处理嵌套括号
        
        Args:
            args_str: 参数字符串，如 "MIN(i0, i1), MAX(i0, i2)"
        
        Returns:
            参数列表
        """
        args = []
        current_arg = ""
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current_arg += char
            elif char == ')':
                depth -= 1
                current_arg += char
            elif char == ',' and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def parse_bench_string(self, content: str):
        """
        解析BENCH格式字符串（支持嵌套表达式）
        
        Args:
            content: BENCH格式的字符串内容
        """
        self.primary_inputs = []
        self.primary_outputs = []
        self.gates = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 解析 INPUT(name)
            input_match = re.match(r'INPUT\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if input_match:
                self.primary_inputs.append(input_match.group(1))
                continue
            
            # 解析 OUTPUT(name)
            output_match = re.match(r'OUTPUT\s*\(\s*(\w+)\s*\)', line, re.IGNORECASE)
            if output_match:
                self.primary_outputs.append(output_match.group(1))
                continue
            
            # 解析 node = EXPR（支持嵌套表达式）
            # 匹配 "node = ..." 形式
            assign_match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if assign_match:
                output_name = assign_match.group(1)
                expr_str = assign_match.group(2)
                
                # 解析表达式
                expr = self._parse_expression(expr_str)
                
                if isinstance(expr, tuple):
                    gate_type, inputs = expr
                    self.gates[output_name] = {
                        "gate_type": gate_type,
                        "inputs": inputs
                    }
                else:
                    # 简单赋值（如 node = input）
                    # 用一个特殊的"BUFFER"门来表示
                    self.gates[output_name] = {
                        "gate_type": "BUFFER",
                        "inputs": [expr]
                    }
    
    def parse_truth_table_file(self, filename: str):
        """
        解析真值表文件（可能包含多个门的定义）
        
        Args:
            filename: 真值表文件路径
        """
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        self.parse_truth_table_string(content)
    
    def parse_truth_table_string(self, content: str):
        """
        解析真值表格式字符串
        
        格式：
        # GATE_NAME num_inputs
        input1,input2,... -> output
        ...
        
        Args:
            content: 真值表格式的字符串内容
        """
        current_gate = None
        current_num_inputs = 0
        current_mapping = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # 跳过空行
            if not line:
                # 如果有当前门，保存它
                if current_gate:
                    self.gate_types[current_gate] = {
                        "name": current_gate,
                        "num_inputs": current_num_inputs,
                        "mapping": current_mapping
                    }
                    current_gate = None
                    current_mapping = {}
                continue
            
            # 解析门定义行: # GATE_NAME num_inputs
            header_match = re.match(r'#\s*(\w+)\s+(\d+)', line)
            if header_match:
                # 保存之前的门（如果有）
                if current_gate:
                    self.gate_types[current_gate] = {
                        "name": current_gate,
                        "num_inputs": current_num_inputs,
                        "mapping": current_mapping
                    }
                
                current_gate = header_match.group(1)
                current_num_inputs = int(header_match.group(2))
                current_mapping = {}
                continue
            
            # 解析映射行: input1,input2,... -> output
            mapping_match = re.match(r'([\d,]+)\s*->\s*(\d+)', line)
            if mapping_match and current_gate:
                inputs_str = mapping_match.group(1)
                output = int(mapping_match.group(2))
                inputs = tuple(int(x) for x in inputs_str.split(','))
                current_mapping[inputs] = output
        
        # 保存最后一个门
        if current_gate:
            self.gate_types[current_gate] = {
                "name": current_gate,
                "num_inputs": current_num_inputs,
                "mapping": current_mapping
            }
    
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
            if expr not in values:
                raise ValueError(f"未知的变量: {expr}")
            return values[expr]
        elif isinstance(expr, tuple):
            # 是嵌套表达式 (gate_type, inputs)
            gate_type, inputs = expr
            
            if gate_type not in self.gate_types:
                raise ValueError(f"未知的门类型: {gate_type}")
            
            # 递归计算每个输入
            input_values = tuple(self._evaluate_expression(inp, values) for inp in inputs)
            
            truth_table = self.gate_types[gate_type]
            if input_values not in truth_table["mapping"]:
                raise ValueError(f"真值表中缺少输入组合: {gate_type}, {input_values}")
            
            return truth_table["mapping"][input_values]
        else:
            raise ValueError(f"未知的表达式类型: {type(expr)}")
    
    def simulate(self, input_values: Dict[str, int]) -> Dict[str, int]:
        """
        仿真电路（支持嵌套表达式）
        
        Args:
            input_values: 主输入的值 {input_name: value}
        
        Returns:
            所有节点的值 {node_name: value}
        """
        values = dict(input_values)
        
        # 对于支持嵌套表达式的电路，直接计算每个门
        for gate_name, gate in self.gates.items():
            gate_type = gate["gate_type"]
            inputs = gate["inputs"]
            
            # 计算每个输入的值（可能是嵌套表达式）
            input_values_list = []
            for inp in inputs:
                input_values_list.append(self._evaluate_expression(inp, values))
            
            input_tuple = tuple(input_values_list)
            
            if gate_type not in self.gate_types:
                raise ValueError(f"未知的门类型: {gate_type}")
            
            truth_table = self.gate_types[gate_type]
            if input_tuple not in truth_table["mapping"]:
                raise ValueError(f"真值表中缺少输入组合: {gate_type}, {input_tuple}")
            
            values[gate_name] = truth_table["mapping"][input_tuple]
        
        return values
    
    def get_output_values(self, input_values: Dict[str, int]) -> List[int]:
        """
        获取所有主输出的值（按顺序）
        
        Args:
            input_values: 主输入的值
        
        Returns:
            主输出的值列表
        """
        all_values = self.simulate(input_values)
        return [all_values[out] for out in self.primary_outputs]
    
    def get_last_output_value(self, input_values: Dict[str, int]) -> int:
        """获取最后一个输出的值"""
        all_values = self.simulate(input_values)
        return all_values[self.primary_outputs[-1]]
    
    def get_other_output_values(self, input_values: Dict[str, int]) -> Tuple[int, ...]:
        """获取除最后一个外的所有输出值（作为元组）"""
        all_values = self.simulate(input_values)
        return tuple(all_values[out] for out in self.primary_outputs[:-1])
    
    def _input_to_string(self, inp) -> str:
        """将输入（可能是字符串或嵌套元组）转换为字符串表示"""
        if isinstance(inp, str):
            return inp
        elif isinstance(inp, tuple):
            gate_type, args = inp
            args_str = ', '.join(self._input_to_string(a) for a in args)
            return f"{gate_type}({args_str})"
        else:
            return str(inp)
    
    def print_info(self):
        """打印电路信息"""
        print("=== 电路信息 ===")
        print(f"主输入: {self.primary_inputs}")
        print(f"主输出: {self.primary_outputs}")
        print(f"门数量: {len(self.gates)}")
        print(f"门类型: {list(self.gate_types.keys())}")
        print()
        print("门列表:")
        for name, gate in self.gates.items():
            inputs_str = ', '.join(self._input_to_string(inp) for inp in gate['inputs'])
            print(f"  {name} = {gate['gate_type']}({inputs_str})")


def load_circuit(bench_file: str, truth_table_file: str) -> BenchParser:
    """
    加载电路（包括BENCH文件和真值表文件）
    
    Args:
        bench_file: BENCH文件路径
        truth_table_file: 真值表文件路径
    
    Returns:
        解析完成的BenchParser对象
    """
    parser = BenchParser()
    parser.parse_bench_file(bench_file)
    parser.parse_truth_table_file(truth_table_file)
    return parser


if __name__ == "__main__":
    # 测试解析器
    
    # 创建测试BENCH内容
    bench_content = """
# 测试电路
INPUT(a)
INPUT(b)
INPUT(c)

OUTPUT(n1)
OUTPUT(n2)
OUTPUT(n3)

n1 = MAX(a, c)
n2 = MAX(b, c)
n3 = MIN(n1, n2)
"""
    
    # 创建测试真值表内容
    truth_table_content = """
# MIN 2
0,0 -> 0
0,1 -> 0
0,2 -> 0
1,0 -> 0
1,1 -> 1
1,2 -> 1
2,0 -> 0
2,1 -> 1
2,2 -> 2

# MAX 2
0,0 -> 0
0,1 -> 1
0,2 -> 2
1,0 -> 1
1,1 -> 1
1,2 -> 2
2,0 -> 2
2,1 -> 2
2,2 -> 2
"""
    
    parser = BenchParser()
    parser.parse_bench_string(bench_content)
    parser.parse_truth_table_string(truth_table_content)
    
    parser.print_info()
    
    # 测试仿真
    print("\n=== 仿真测试 ===")
    test_inputs = [
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 0},
        {"a": 1, "b": 0, "c": 1},
        {"a": 2, "b": 2, "c": 2},
    ]
    
    for inp in test_inputs:
        outputs = parser.get_output_values(inp)
        print(f"输入: {inp}")
        print(f"输出: n1={outputs[0]}, n2={outputs[1]}, n3={outputs[2]}")
        print()
