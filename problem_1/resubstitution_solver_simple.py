"""
三值逻辑重替换求解器 - 简洁版（约60行核心代码）
适合比赛手写，易记易背

【核心思想】
判断最后输出能否由其他输出表示 ⟺ 不存在两组输入使得其他输出相同但最后输出不同

【算法】
1. 小规模（n≤10）：直接枚举所有3^n个输入
2. 大规模：随机采样检测冲突（实践中足够）
"""

import itertools
import random


def simulate(circuit, inputs):
    """模拟电路，返回所有输出值"""
    vals = dict(inputs)
    gates = circuit['gates']
    gate_types = circuit['gate_types']
    
    def eval_expr(expr):
        if isinstance(expr, str):
            return vals[expr] if expr in vals else vals.setdefault(expr, eval_gate(expr))
        gate_name, args = expr
        arg_vals = [eval_expr(a) for a in args]
        return gate_types[gate_name][tuple(arg_vals)]
    
    def eval_gate(name):
        gate_name, args = gates[name]
        arg_vals = [eval_expr(a) for a in args]
        return gate_types[gate_name][tuple(arg_vals)]
    
    return {out: eval_expr(gates[out]) for out in circuit['outputs']}


def solve(circuit, max_samples=100000):
    """
    判断最后输出能否由其他输出表示
    
    Returns: (can_resub, function_table or conflict)
        can_resub: True/False
        function_table: {other_outputs_tuple: last_output} 或 冲突信息
    """
    inputs = circuit['inputs']
    outputs = circuit['outputs']
    n = len(inputs)
    
    # mapping: other_outputs -> (last_output, input_tuple)
    mapping = {}
    
    def check(input_vals):
        """检查一组输入，返回冲突信息或None"""
        inp_dict = dict(zip(inputs, input_vals))
        out_vals = simulate(circuit, inp_dict)
        
        other = tuple(out_vals[o] for o in outputs[:-1])
        last = out_vals[outputs[-1]]
        
        if other in mapping:
            if mapping[other][0] != last:
                return (other, mapping[other][0], last, mapping[other][1], input_vals)
        else:
            mapping[other] = (last, input_vals)
        return None
    
    # 小规模：完全枚举
    if 3**n <= max_samples:
        for inp in itertools.product(range(3), repeat=n):
            conflict = check(inp)
            if conflict:
                return False, conflict
    else:
        # 大规模：随机采样
        # 先试结构化向量
        for inp in [(0,)*n, (1,)*n, (2,)*n]:
            check(inp)
        
        # 随机采样
        for _ in range(max_samples):
            inp = tuple(random.randint(0,2) for _ in range(n))
            conflict = check(inp)
            if conflict:
                return False, conflict
    
    # 无冲突，构建函数表
    func_table = {k: v[0] for k, v in mapping.items()}
    return True, func_table


# ========== 简易测试 ==========
if __name__ == "__main__":
    # 构造一个简单电路测试
    # 门定义：三值逻辑真值表
    MIN_TABLE = {(a,b): min(a,b) for a in range(3) for b in range(3)}
    MAX_TABLE = {(a,b): max(a,b) for a in range(3) for b in range(3)}
    
    # 正例：o2 = MIN(o0, o1) 展开后
    circuit_pos = {
        'inputs': ['i0', 'i1', 'i2'],
        'outputs': ['o0', 'o1', 'o2'],
        'gates': {
            'o0': ('MIN', ['i0', 'i1']),
            'o1': ('MAX', ['i0', 'i2']),
            'o2': ('MIN', [('MIN', ['i0', 'i1']), ('MAX', ['i0', 'i2'])])
        },
        'gate_types': {'MIN': MIN_TABLE, 'MAX': MAX_TABLE}
    }
    
    # 反例：o2 依赖独立输入 i2
    circuit_neg = {
        'inputs': ['i0', 'i1', 'i2'],
        'outputs': ['o0', 'o1', 'o2'],
        'gates': {
            'o0': ('MIN', ['i0', 'i1']),
            'o1': ('MAX', ['i0', 'i1']),
            'o2': ('MIN', ['i0', 'i2'])  # i2 不影响 o0, o1
        },
        'gate_types': {'MIN': MIN_TABLE, 'MAX': MAX_TABLE}
    }
    
    print("=== 正例测试 ===")
    can, result = solve(circuit_pos)
    print(f"可替换: {can}")
    if can:
        print(f"函数表大小: {len(result)}")
    
    print("\n=== 反例测试 ===")
    can, result = solve(circuit_neg)
    print(f"可替换: {can}")
    if not can:
        print(f"冲突: other={result[0]}, last可为{result[1]}或{result[2]}")
