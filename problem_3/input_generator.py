"""
输入生成器：生成模拟电路参数表达式的测试用例

生成的表达式格式：
- 赋值语句：变量名 = 表达式
- 支持的运算：+, -, *, /, %, **（幂运算）
- 支持的函数：sin, cos, tan, sqrt, abs, log, exp
- 支持条件表达式：condition ? true_value : false_value
- 变量可以依赖其他变量，形成 DAG 结构
"""

import random
import math
from typing import List, Dict, Tuple, Set


class ExpressionGenerator:
    """表达式生成器：生成具有依赖关系的电路参数表达式"""
    
    def __init__(self, seed: int = None):
        """
        初始化生成器
        :param seed: 随机种子（可选，用于复现测试用例）
        """
        if seed is not None:
            random.seed(seed)
        
        # 支持的二元运算符
        self.binary_ops = ['+', '-', '*', '/']
        # 支持的数学函数（移除 exp 以避免溢出）
        self.functions = ['sin', 'cos', 'sqrt', 'abs']
        # 比较运算符（用于条件表达式）
        self.compare_ops = ['>', '<', '>=', '<=', '==', '!=']
    
    def generate_test_case(self,
                           num_base_vars: int = 5,
                           num_derived_vars: int = 20,
                           max_deps_per_var: int = 3,
                           ternary_prob: float = 0.15,
                           function_prob: float = 0.2) -> Dict:
        """
        生成一个完整的测试用例
        
        :param num_base_vars: 基础变量数量（叶节点，无依赖）
        :param num_derived_vars: 派生变量数量（依赖于其他变量）
        :param max_deps_per_var: 每个派生变量最多依赖的变量数
        :param ternary_prob: 使用三元表达式的概率
        :param function_prob: 使用函数调用的概率
        :return: 测试用例字典，包含表达式、初始值、变更信息等
        """
        expressions = {}  # 变量名 -> 表达式字符串
        base_values = {}  # 基础变量 -> 初始值
        
        # 1. 生成基础变量（叶节点）
        base_vars = [f"var{i}" for i in range(1, num_base_vars + 1)]
        for var in base_vars:
            # 为基础变量分配随机初始值
            base_values[var] = round(random.uniform(0.1, 10.0), 3)
            expressions[var] = str(base_values[var])  # 基础变量的"表达式"就是其值
        
        # 2. 生成派生变量（按顺序，确保依赖的变量已定义）
        all_vars = base_vars.copy()  # 当前可用于依赖的变量列表
        derived_vars = []
        
        for i in range(num_derived_vars):
            var_name = f"node{i}"
            derived_vars.append(var_name)
            
            # 随机选择该变量依赖的变量（1到max_deps个）
            num_deps = random.randint(1, min(max_deps_per_var, len(all_vars)))
            deps = random.sample(all_vars, num_deps)
            
            # 生成表达式
            expr = self._generate_expression(deps, ternary_prob, function_prob)
            expressions[var_name] = expr
            
            # 将新变量加入可用列表
            all_vars.append(var_name)
        
        # 3. 生成一些"变更场景"（模拟参数更新）
        changes = self._generate_changes(base_vars, base_values)
        
        return {
            "expressions": expressions,
            "base_vars": base_vars,
            "derived_vars": derived_vars,
            "base_values": base_values,
            "changes": changes
        }
    
    def _generate_expression(self, deps: List[str], 
                            ternary_prob: float,
                            function_prob: float) -> str:
        """
        生成一个表达式
        
        :param deps: 该表达式依赖的变量列表
        :param ternary_prob: 使用三元表达式的概率
        :param function_prob: 使用函数调用的概率
        :return: 表达式字符串
        """
        # 决定表达式类型
        rand_val = random.random()
        
        if rand_val < ternary_prob and len(deps) >= 2:
            # 生成三元条件表达式：cond ? val1 : val2
            return self._generate_ternary(deps)
        elif rand_val < ternary_prob + function_prob:
            # 生成函数调用表达式
            return self._generate_function_call(deps)
        else:
            # 生成普通算术表达式
            return self._generate_arithmetic(deps)
    
    def _generate_arithmetic(self, deps: List[str]) -> str:
        """生成算术表达式"""
        if len(deps) == 1:
            # 单个依赖：可能加常数或乘常数
            var = deps[0]
            op = random.choice(self.binary_ops)
            const = round(random.uniform(0.5, 5.0), 2)
            if random.random() < 0.5:
                return f"{var} {op} {const}"
            else:
                return f"{const} {op} {var}" if op in ['+', '*'] else f"{var} {op} {const}"
        else:
            # 多个依赖：组合它们
            expr_parts = []
            for var in deps:
                if random.random() < 0.3:
                    # 有时给变量乘一个系数
                    coef = round(random.uniform(0.5, 2.0), 2)
                    expr_parts.append(f"{coef} * {var}")
                else:
                    expr_parts.append(var)
            
            # 用运算符连接
            result = expr_parts[0]
            for part in expr_parts[1:]:
                op = random.choice(self.binary_ops)
                result = f"({result} {op} {part})"
            return result
    
    def _generate_function_call(self, deps: List[str]) -> str:
        """生成函数调用表达式"""
        func = random.choice(self.functions)
        var = random.choice(deps)
        
        # 对于某些函数，需要确保参数合理
        if func in ['sqrt', 'log']:
            # 取绝对值确保非负
            inner = f"abs({var})"
        else:
            inner = var
        
        # 有时添加额外运算
        if len(deps) > 1 and random.random() < 0.5:
            other_var = random.choice([v for v in deps if v != var])
            op = random.choice(['+', '*'])
            return f"{func}({inner}) {op} {other_var}"
        else:
            return f"{func}({inner})"
    
    def _generate_ternary(self, deps: List[str]) -> str:
        """生成三元条件表达式"""
        # 选择用于条件判断的变量
        cond_var = random.choice(deps)
        compare_op = random.choice(self.compare_ops)
        threshold = round(random.uniform(1.0, 5.0), 2)
        
        # 条件部分
        condition = f"{cond_var} {compare_op} {threshold}"
        
        # true/false 分支
        if len(deps) >= 2:
            true_val = deps[0]
            false_val = deps[1]
        else:
            true_val = deps[0]
            false_val = str(round(random.uniform(0.1, 5.0), 2))
        
        return f"({condition}) ? {true_val} : {false_val}"
    
    def _generate_changes(self, base_vars: List[str], 
                         base_values: Dict[str, float]) -> List[Dict]:
        """
        生成变更场景列表
        
        :param base_vars: 基础变量列表
        :param base_values: 基础变量的初始值
        :return: 变更场景列表
        """
        changes = []
        
        # 场景1：单个变量变更
        for i in range(min(3, len(base_vars))):
            var = base_vars[i]
            new_val = round(base_values[var] * random.uniform(0.5, 2.0), 3)
            changes.append({
                "description": f"修改 {var}",
                "changed_vars": {var: new_val}
            })
        
        # 场景2：多个变量同时变更
        if len(base_vars) >= 2:
            vars_to_change = random.sample(base_vars, min(3, len(base_vars)))
            changed = {}
            for var in vars_to_change:
                changed[var] = round(base_values[var] * random.uniform(0.5, 2.0), 3)
            changes.append({
                "description": f"同时修改 {', '.join(vars_to_change)}",
                "changed_vars": changed
            })
        
        return changes


def generate_simple_case() -> Dict:
    """生成简单测试用例（题目示例）"""
    return {
        "expressions": {
            "var1": "5.0",  # 基础变量
            "A": "var1 + 2",
            "B": "A / 2",
            "C": "(B > 1.0) ? var1 : A"
        },
        "base_vars": ["var1"],
        "derived_vars": ["A", "B", "C"],
        "base_values": {"var1": 5.0},
        "changes": [
            {"description": "修改 var1 为 3.0", "changed_vars": {"var1": 3.0}},
            {"description": "修改 var1 为 0.5", "changed_vars": {"var1": 0.5}}
        ]
    }


def generate_medium_case() -> Dict:
    """生成中等规模测试用例"""
    gen = ExpressionGenerator(seed=42)
    return gen.generate_test_case(
        num_base_vars=5,
        num_derived_vars=15,
        max_deps_per_var=3,
        ternary_prob=0.15,
        function_prob=0.2
    )


def generate_large_case() -> Dict:
    """生成大规模测试用例"""
    gen = ExpressionGenerator(seed=123)
    return gen.generate_test_case(
        num_base_vars=10,
        num_derived_vars=100,
        max_deps_per_var=5,
        ternary_prob=0.1,
        function_prob=0.15
    )


def generate_multi_dag_case() -> Dict:
    """生成包含多个独立 DAG 的测试用例"""
    return {
        "expressions": {
            # DAG 1: var1 -> A -> B -> C
            "var1": "1.0",
            "A": "var1 + 2",
            "B": "A * 2",
            "C": "B - 1",
            # DAG 2: var2 -> X -> Y
            "var2": "3.0",
            "X": "var2 * 3",
            "Y": "X + 5",
            # DAG 3: var3 -> P -> Q (with shared dependency)
            "var3": "2.0",
            "P": "var3 / 2",
            "Q": "P * P"
        },
        "base_vars": ["var1", "var2", "var3"],
        "derived_vars": ["A", "B", "C", "X", "Y", "P", "Q"],
        "base_values": {"var1": 1.0, "var2": 3.0, "var3": 2.0},
        "changes": [
            {"description": "修改 var1（只影响 DAG1）", "changed_vars": {"var1": 5.0}},
            {"description": "修改 var2（只影响 DAG2）", "changed_vars": {"var2": 10.0}},
            {"description": "同时修改 var1 和 var3", "changed_vars": {"var1": 2.0, "var3": 4.0}}
        ]
    }


def save_test_case(test_case: Dict, filename: str):
    """保存测试用例到文件"""
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(test_case, f, indent=2, ensure_ascii=False)
    print(f"测试用例已保存到: {filename}")


def load_test_case(filename: str) -> Dict:
    """从文件加载测试用例"""
    import json
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import os
    
    # 创建测试用例目录
    test_dir = os.path.join(os.path.dirname(__file__), "test_cases")
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成并保存各种测试用例
    print("=" * 50)
    print("生成测试用例")
    print("=" * 50)
    
    # 简单用例
    simple = generate_simple_case()
    save_test_case(simple, os.path.join(test_dir, "simple.json"))
    print(f"简单用例: {len(simple['expressions'])} 个表达式")
    
    # 中等用例
    medium = generate_medium_case()
    save_test_case(medium, os.path.join(test_dir, "medium.json"))
    print(f"中等用例: {len(medium['expressions'])} 个表达式")
    
    # 大规模用例
    large = generate_large_case()
    save_test_case(large, os.path.join(test_dir, "large.json"))
    print(f"大规模用例: {len(large['expressions'])} 个表达式")
    
    # 多 DAG 用例
    multi_dag = generate_multi_dag_case()
    save_test_case(multi_dag, os.path.join(test_dir, "multi_dag.json"))
    print(f"多DAG用例: {len(multi_dag['expressions'])} 个表达式")
    
    print("\n所有测试用例生成完毕！")
