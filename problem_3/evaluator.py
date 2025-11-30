"""
增量表达式求值器

核心功能：
1. 首次完整求值：按拓扑序计算所有表达式
2. 增量求值：当部分变量变化时，只重算受影响的表达式
3. 缓存管理：缓存计算结果，避免重复计算
4. 支持多种运算：算术、函数调用、条件表达式

算法核心思想：
- 利用 DAG 的拓扑结构，保证计算顺序正确
- 通过脏节点传播，精确识别需要重算的节点
- 只对脏节点子图进行局部拓扑排序，避免全图排序开销
"""

import math
import re
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from expression_parser import DependencyGraph, DirtyNodeTracker, build_dependency_graph


class ExpressionEvaluator:
    """
    表达式求值器
    
    将表达式字符串求值为数值结果
    支持：
    - 基本算术：+, -, *, /, %, **
    - 数学函数：sin, cos, tan, sqrt, abs, log, exp, pow, min, max
    - 三元表达式：condition ? true_val : false_val
    - 比较运算：>, <, >=, <=, ==, !=
    """
    
    # 三元表达式的正则模式
    TERNARY_PATTERN = re.compile(r'\(([^?]+)\)\s*\?\s*([^:]+)\s*:\s*(.+)')
    
    def __init__(self):
        """初始化求值器"""
        # 内置函数映射
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'abs': abs,
            'log': math.log,
            'exp': math.exp,
            'pow': pow,
            'min': min,
            'max': max,
            'floor': math.floor,
            'ceil': math.ceil,
            'round': round,
        }
    
    def evaluate(self, expression: str, context: Dict[str, float]) -> float:
        """
        求值单个表达式
        
        :param expression: 表达式字符串
        :param context: 变量名 -> 值 的上下文字典
        :return: 计算结果
        """
        # 首先检查是否是三元表达式
        ternary_match = self.TERNARY_PATTERN.match(expression.strip())
        if ternary_match:
            return self._evaluate_ternary(ternary_match, context)
        
        # 普通表达式求值
        return self._evaluate_expression(expression, context)
    
    def _evaluate_ternary(self, match: re.Match, context: Dict[str, float]) -> float:
        """
        求值三元表达式
        
        :param match: 三元表达式的正则匹配结果
        :param context: 变量上下文
        :return: 计算结果
        """
        condition_str = match.group(1).strip()
        true_val_str = match.group(2).strip()
        false_val_str = match.group(3).strip()
        
        # 计算条件
        condition = self._evaluate_expression(condition_str, context)
        
        # 根据条件选择分支
        if condition:
            return self._evaluate_expression(true_val_str, context)
        else:
            return self._evaluate_expression(false_val_str, context)
    
    def _evaluate_expression(self, expression: str, context: Dict[str, float]) -> float:
        """
        求值普通表达式（使用 Python 的 eval）
        
        :param expression: 表达式字符串
        :param context: 变量上下文
        :return: 计算结果
        """
        # 构建安全的求值环境
        safe_env = {
            '__builtins__': {},  # 禁用内置函数
            **self.functions,     # 添加数学函数
            **context             # 添加变量值
        }
        
        try:
            result = eval(expression, safe_env)
            return float(result)
        except Exception as e:
            raise ValueError(f"表达式求值失败: {expression}, 错误: {e}")


class IncrementalEvaluationEngine:
    """
    增量求值引擎
    
    这是整个系统的核心，负责：
    1. 管理表达式和依赖图
    2. 缓存计算结果
    3. 跟踪值的版本号（用于检测变化）
    4. 执行增量求值
    
    算法流程（增量求值）：
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. 接收变化的源节点列表                                      │
    │ 2. 脏标记传播：BFS 标记所有受影响的节点                       │
    │ 3. 局部拓扑排序：只对脏节点子图排序                           │
    │ 4. 按序求值：依次计算脏节点，更新缓存                         │
    │ 5. 返回更新后的结果                                          │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, expressions: Dict[str, str]):
        """
        初始化引擎
        
        :param expressions: 变量名 -> 表达式字符串 的字典
        """
        self.expressions = expressions.copy()
        
        # 构建依赖图
        self.graph = build_dependency_graph(expressions)
        
        # 创建脏节点追踪器
        self.tracker = DirtyNodeTracker(self.graph)
        
        # 创建表达式求值器
        self.evaluator = ExpressionEvaluator()
        
        # 值缓存：存储每个变量的当前值
        self.value_cache: Dict[str, float] = {}
        
        # 版本号：用于跟踪值的变化
        self.version: Dict[str, int] = {node: 0 for node in self.graph.nodes}
        
        # 是否已初始化（是否已进行首次完整求值）
        self._initialized = False
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,    # 总求值次数
            'incremental_updates': 0,  # 增量更新次数
            'nodes_recomputed': 0      # 重算的节点数
        }
    
    def initialize(self, base_values: Dict[str, float] = None):
        """
        初始化：进行首次完整求值
        
        :param base_values: 基础变量的初始值（可选）
        """
        # 设置基础变量的值
        if base_values:
            for var, value in base_values.items():
                self.value_cache[var] = value
        
        # 按拓扑序计算所有表达式
        topo_order = self.graph.get_topological_order()
        
        for node in topo_order:
            if node not in self.value_cache:
                # 需要计算这个节点
                self._compute_node(node)
        
        self._initialized = True
        self.stats['total_evaluations'] = len(topo_order)
    
    def update(self, changed_vars: Dict[str, float]) -> Dict[str, Any]:
        """
        增量更新：当某些变量值改变时，重新计算受影响的表达式
        
        这是增量求值的核心方法！
        
        算法步骤：
        1. 更新变化的源节点值
        2. 使用脏节点追踪器找出所有需要重算的节点
        3. 按拓扑序重新计算这些节点
        4. 返回更新结果和统计信息
        
        :param changed_vars: 变化的变量及其新值
        :return: 更新结果字典
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用 initialize()")
        
        # 记录旧值（用于比较）
        old_values = {node: self.value_cache.get(node) for node in self.graph.nodes}
        
        # 步骤1：更新变化的源节点
        changed_sources = []
        for var, new_value in changed_vars.items():
            if var in self.graph.nodes:
                old_value = self.value_cache.get(var)
                if old_value != new_value:
                    self.value_cache[var] = new_value
                    self.version[var] += 1
                    changed_sources.append(var)
        
        if not changed_sources:
            return {
                'changed_sources': [],
                'dirty_nodes': set(),
                'recompute_order': [],
                'updated_values': {},
                'stats': {'nodes_recomputed': 0}
            }
        
        # 步骤2：找出所有脏节点及其重算顺序
        dirty_nodes, recompute_order = self.tracker.get_recompute_order(changed_sources)
        
        # 步骤3：按拓扑序重新计算脏节点
        updated_values = {}
        for node in recompute_order:
            if node not in changed_vars:  # 源节点已更新，跳过
                old_val = self.value_cache.get(node)
                self._compute_node(node)
                new_val = self.value_cache[node]
                
                # 检测值是否真的变化了
                if old_val != new_val:
                    self.version[node] += 1
                    updated_values[node] = {
                        'old': old_val,
                        'new': new_val
                    }
        
        # 更新统计
        self.stats['incremental_updates'] += 1
        self.stats['nodes_recomputed'] += len(recompute_order)
        self.stats['total_evaluations'] += len(recompute_order)
        
        return {
            'changed_sources': changed_sources,
            'dirty_nodes': dirty_nodes,
            'recompute_order': recompute_order,
            'updated_values': updated_values,
            'stats': {
                'nodes_recomputed': len(recompute_order),
                'total_nodes': len(self.graph.nodes)
            }
        }
    
    def _compute_node(self, node: str):
        """
        计算单个节点的值
        
        :param node: 节点名称
        """
        expr = self.expressions.get(node, str(node))
        
        # 尝试直接解析为数值（常量）
        try:
            value = float(expr)
            self.value_cache[node] = value
            return
        except ValueError:
            pass
        
        # 需要求值表达式
        value = self.evaluator.evaluate(expr, self.value_cache)
        self.value_cache[node] = value
    
    def get_value(self, var: str) -> Optional[float]:
        """获取变量的当前值"""
        return self.value_cache.get(var)
    
    def get_all_values(self) -> Dict[str, float]:
        """获取所有变量的当前值"""
        return self.value_cache.copy()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def print_state(self):
        """打印当前状态（调试用）"""
        print("\n当前状态:")
        print("-" * 40)
        topo = self.graph.get_topological_order()
        for node in topo:
            expr = self.expressions.get(node, "N/A")
            value = self.value_cache.get(node, "N/A")
            version = self.version.get(node, 0)
            print(f"  {node} = {expr}")
            print(f"       值: {value}, 版本: {version}")


class BatchEvaluator:
    """
    批量求值器
    
    针对参数扫描场景优化：
    当需要对多组参数值进行求值时，可以复用共享的计算结果
    """
    
    def __init__(self, expressions: Dict[str, str]):
        """
        初始化批量求值器
        
        :param expressions: 变量名 -> 表达式字符串
        """
        self.expressions = expressions
        self.graph = build_dependency_graph(expressions)
        self.evaluator = ExpressionEvaluator()
    
    def evaluate_batch(self, 
                       base_var: str,
                       values: List[float]) -> List[Dict[str, float]]:
        """
        对单个变量的多个值进行批量求值
        
        :param base_var: 要变化的基础变量名
        :param values: 该变量的值列表
        :return: 每组值对应的完整结果字典列表
        """
        results = []
        engine = IncrementalEvaluationEngine(self.expressions)
        
        # 首次初始化
        first_values = {base_var: values[0]}
        engine.initialize(first_values)
        results.append(engine.get_all_values())
        
        # 增量更新后续值
        for val in values[1:]:
            engine.update({base_var: val})
            results.append(engine.get_all_values())
        
        return results


def compare_full_vs_incremental(expressions: Dict[str, str],
                                 base_values: Dict[str, float],
                                 changes: List[Dict[str, float]]) -> Dict:
    """
    比较完整求值 vs 增量求值的性能
    
    :param expressions: 表达式字典
    :param base_values: 初始值
    :param changes: 变化列表
    :return: 比较结果
    """
    import time
    
    # 方法1：每次都完整求值
    full_eval_times = []
    for change in changes:
        merged_values = {**base_values, **change}
        engine = IncrementalEvaluationEngine(expressions)
        
        start = time.perf_counter()
        engine.initialize(merged_values)
        end = time.perf_counter()
        
        full_eval_times.append(end - start)
    
    # 方法2：增量求值
    engine = IncrementalEvaluationEngine(expressions)
    engine.initialize(base_values)
    
    incr_eval_times = []
    for change in changes:
        start = time.perf_counter()
        engine.update(change)
        end = time.perf_counter()
        
        incr_eval_times.append(end - start)
    
    return {
        'full_eval_times': full_eval_times,
        'incremental_eval_times': incr_eval_times,
        'full_total': sum(full_eval_times),
        'incremental_total': sum(incr_eval_times),
        'speedup': sum(full_eval_times) / max(sum(incr_eval_times), 1e-9)
    }


if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("增量表达式求值器测试")
    print("=" * 60)
    
    # 题目示例
    expressions = {
        "var1": "5.0",
        "A": "var1 + 2",
        "B": "A / 2",
        "C": "(B > 1.0) ? var1 : A"
    }
    
    print("\n表达式定义:")
    for var, expr in expressions.items():
        print(f"  {var} = {expr}")
    
    # 创建引擎
    engine = IncrementalEvaluationEngine(expressions)
    
    # 初始化
    print("\n--- 初始化 ---")
    engine.initialize({"var1": 5.0})
    engine.print_state()
    
    # 增量更新测试1
    print("\n--- 增量更新: var1 = 3.0 ---")
    result = engine.update({"var1": 3.0})
    print(f"变化源: {result['changed_sources']}")
    print(f"脏节点: {result['dirty_nodes']}")
    print(f"重算顺序: {result['recompute_order']}")
    print(f"重算节点数: {result['stats']['nodes_recomputed']}/{result['stats']['total_nodes']}")
    engine.print_state()
    
    # 增量更新测试2
    print("\n--- 增量更新: var1 = 0.5 ---")
    result = engine.update({"var1": 0.5})
    print(f"变化源: {result['changed_sources']}")
    print(f"脏节点: {result['dirty_nodes']}")
    print(f"重算顺序: {result['recompute_order']}")
    engine.print_state()
    
    print("\n" + "=" * 60)
    print("统计信息:")
    stats = engine.get_stats()
    print(f"  总求值次数: {stats['total_evaluations']}")
    print(f"  增量更新次数: {stats['incremental_updates']}")
    print(f"  重算节点总数: {stats['nodes_recomputed']}")
