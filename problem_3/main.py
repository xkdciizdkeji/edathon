"""
Problem 3: Efficient Expressions Evaluation
高效表达式求值 - 主程序

本程序实现了一个高效的增量表达式求值系统，用于处理电路仿真中的参数依赖关系。

核心思想：
1. 将表达式之间的依赖关系建模为有向无环图（DAG）
2. 当部分参数变化时，只重新计算受影响的节点（脏节点）
3. 通过局部拓扑排序确保正确的计算顺序

主要组件：
- input_generator.py: 生成测试用例
- expression_parser.py: 解析表达式，构建依赖图
- evaluator.py: 增量求值引擎

使用方法：
    python main.py [test_case_name]
    
    test_case_name 可选值:
    - simple: 题目示例
    - medium: 中等规模
    - large: 大规模
    - multi_dag: 多DAG测试
"""

import os
import sys
import json
import time
from typing import Dict, List, Any

# 导入自定义模块
from input_generator import (
    generate_simple_case, 
    generate_medium_case, 
    generate_large_case,
    generate_multi_dag_case,
    save_test_case,
    load_test_case
)
from expression_parser import (
    build_dependency_graph,
    DirtyNodeTracker,
    DependencyGraph
)
from evaluator import (
    IncrementalEvaluationEngine,
    compare_full_vs_incremental
)


def print_header(title: str):
    """打印格式化标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """打印小节标题"""
    print(f"\n--- {title} ---")


def visualize_dag(graph: DependencyGraph, 
                  dirty_nodes: set = None,
                  max_nodes: int = 20) -> str:
    """
    简单可视化 DAG 结构
    
    :param graph: 依赖图
    :param dirty_nodes: 脏节点集合（高亮显示）
    :param max_nodes: 最多显示的节点数
    :return: 可视化字符串
    """
    lines = []
    topo = graph.get_topological_order()
    
    if len(topo) > max_nodes:
        lines.append(f"(仅显示前 {max_nodes} 个节点，共 {len(topo)} 个)")
        topo = topo[:max_nodes]
    
    for node in topo:
        deps = graph.adj_in[node]
        affected = graph.adj_out[node]
        
        # 标记脏节点
        marker = " [脏]" if dirty_nodes and node in dirty_nodes else ""
        
        if deps:
            dep_str = ", ".join(deps)
            lines.append(f"  {node}{marker} <- [{dep_str}]")
        else:
            lines.append(f"  {node}{marker} (基础节点)")
    
    return "\n".join(lines)


def run_test_case(test_case: Dict, verbose: bool = True) -> Dict:
    """
    运行单个测试用例
    
    :param test_case: 测试用例字典
    :param verbose: 是否输出详细信息
    :return: 测试结果
    """
    expressions = test_case['expressions']
    base_values = test_case.get('base_values', {})
    changes = test_case.get('changes', [])
    
    results = {
        'num_expressions': len(expressions),
        'num_changes': len(changes),
        'change_results': [],
        'timing': {}
    }
    
    # 1. 构建依赖图
    if verbose:
        print_section("构建依赖图")
    
    start_time = time.perf_counter()
    graph = build_dependency_graph(expressions)
    build_time = time.perf_counter() - start_time
    results['timing']['build_graph'] = build_time
    
    if verbose:
        print(f"节点数: {len(graph.nodes)}")
        print(f"基础节点: {graph.base_nodes}")
        print(f"构建时间: {build_time*1000:.3f} ms")
    
    # 2. 检测循环依赖
    cycles = graph.detect_cycles()
    if cycles:
        print(f"警告：检测到循环依赖: {cycles}")
        results['has_cycles'] = True
        return results
    results['has_cycles'] = False
    
    # 3. 显示 DAG 结构
    if verbose and len(graph.nodes) <= 20:
        print_section("DAG 结构")
        print(visualize_dag(graph))
    
    # 4. 初始化求值引擎
    if verbose:
        print_section("初始化求值")
    
    engine = IncrementalEvaluationEngine(expressions)
    
    start_time = time.perf_counter()
    engine.initialize(base_values)
    init_time = time.perf_counter() - start_time
    results['timing']['initialize'] = init_time
    
    if verbose:
        print(f"初始化时间: {init_time*1000:.3f} ms")
        if len(expressions) <= 10:
            print("\n初始值:")
            for var in graph.get_topological_order():
                val = engine.get_value(var)
                print(f"  {var} = {val:.4f}" if val is not None else f"  {var} = N/A")
    
    # 5. 执行增量更新
    if changes and verbose:
        print_section("增量更新测试")
    
    for i, change in enumerate(changes):
        changed_vars = change.get('changed_vars', {})
        description = change.get('description', f"变更 {i+1}")
        
        if verbose:
            print(f"\n[{i+1}] {description}")
            print(f"    变更内容: {changed_vars}")
        
        start_time = time.perf_counter()
        update_result = engine.update(changed_vars)
        update_time = time.perf_counter() - start_time
        
        change_result = {
            'description': description,
            'changed_vars': changed_vars,
            'dirty_nodes': list(update_result['dirty_nodes']),
            'recompute_order': update_result['recompute_order'],
            'nodes_recomputed': update_result['stats']['nodes_recomputed'],
            'total_nodes': update_result['stats']['total_nodes'],
            'time_ms': update_time * 1000
        }
        results['change_results'].append(change_result)
        
        if verbose:
            dirty_count = len(update_result['dirty_nodes'])
            total_count = update_result['stats']['total_nodes']
            print(f"    脏节点数: {dirty_count}/{total_count} ({dirty_count/total_count*100:.1f}%)")
            print(f"    重算顺序: {update_result['recompute_order'][:10]}{'...' if len(update_result['recompute_order']) > 10 else ''}")
            print(f"    更新时间: {update_time*1000:.3f} ms")
            
            # 显示更新后的关键值
            if len(expressions) <= 10:
                print(f"    更新后的值:")
                for var in update_result['recompute_order'][:5]:
                    val = engine.get_value(var)
                    print(f"      {var} = {val:.4f}")
    
    # 6. 统计信息
    if verbose:
        print_section("统计汇总")
        stats = engine.get_stats()
        print(f"总求值次数: {stats['total_evaluations']}")
        print(f"增量更新次数: {stats['incremental_updates']}")
        print(f"重算节点总数: {stats['nodes_recomputed']}")
        
        # 计算节省的计算量
        if changes:
            full_evals = len(expressions) * len(changes)
            actual_evals = stats['nodes_recomputed']
            saved = (1 - actual_evals / full_evals) * 100 if full_evals > 0 else 0
            print(f"\n与完整求值对比:")
            print(f"  完整求值需要: {full_evals} 次节点计算")
            print(f"  增量求值实际: {actual_evals} 次节点计算")
            print(f"  节省计算量: {saved:.1f}%")
    
    return results


def benchmark_performance(sizes: List[int] = [10, 50, 100, 200, 500]) -> Dict:
    """
    性能基准测试
    
    :param sizes: 测试的表达式数量列表
    :return: 基准测试结果
    """
    from input_generator import ExpressionGenerator
    
    print_header("性能基准测试")
    
    results = []
    
    for size in sizes:
        print(f"\n测试规模: {size} 个表达式")
        
        # 生成测试用例
        gen = ExpressionGenerator(seed=size)
        num_base = max(3, size // 10)
        num_derived = size - num_base
        
        test_case = gen.generate_test_case(
            num_base_vars=num_base,
            num_derived_vars=num_derived,
            max_deps_per_var=min(5, num_base)
        )
        
        # 运行测试
        expressions = test_case['expressions']
        base_values = test_case['base_values']
        
        # 构建图
        start = time.perf_counter()
        graph = build_dependency_graph(expressions)
        build_time = (time.perf_counter() - start) * 1000
        
        # 初始化
        engine = IncrementalEvaluationEngine(expressions)
        start = time.perf_counter()
        engine.initialize(base_values)
        init_time = (time.perf_counter() - start) * 1000
        
        # 单变量更新测试
        base_var = list(base_values.keys())[0]
        times_single = []
        for _ in range(5):
            start = time.perf_counter()
            engine.update({base_var: 999.0})
            times_single.append((time.perf_counter() - start) * 1000)
        avg_single = sum(times_single) / len(times_single)
        
        result = {
            'size': size,
            'build_time_ms': build_time,
            'init_time_ms': init_time,
            'update_time_ms': avg_single
        }
        results.append(result)
        
        print(f"  构建图: {build_time:.3f} ms")
        print(f"  初始化: {init_time:.3f} ms")
        print(f"  增量更新: {avg_single:.3f} ms (平均)")
    
    return results


def main():
    """主函数"""
    print_header("Problem 3: Efficient Expressions Evaluation")
    print("高效表达式求值系统")
    print("\n本系统通过 DAG 建模和增量计算，实现高效的表达式求值。")
    print("当参数变化时，只重新计算受影响的表达式，避免重复计算。")
    
    # 确定运行哪个测试用例
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
    else:
        test_name = "simple"
    
    # 生成或加载测试用例
    test_dir = os.path.join(os.path.dirname(__file__), "test_cases")
    os.makedirs(test_dir, exist_ok=True)
    
    if test_name == "simple":
        print_header("测试用例: 题目示例 (Simple)")
        test_case = generate_simple_case()
    elif test_name == "medium":
        print_header("测试用例: 中等规模 (Medium)")
        test_case = generate_medium_case()
    elif test_name == "large":
        print_header("测试用例: 大规模 (Large)")
        test_case = generate_large_case()
    elif test_name == "multi_dag":
        print_header("测试用例: 多 DAG (Multi-DAG)")
        test_case = generate_multi_dag_case()
    elif test_name == "benchmark":
        benchmark_performance()
        return
    else:
        # 尝试从文件加载
        test_file = os.path.join(test_dir, f"{test_name}.json")
        if os.path.exists(test_file):
            print_header(f"测试用例: {test_name}")
            test_case = load_test_case(test_file)
        else:
            print(f"未知的测试用例: {test_name}")
            print("可用选项: simple, medium, large, multi_dag, benchmark")
            return
    
    # 显示表达式
    print_section("表达式定义")
    expressions = test_case['expressions']
    if len(expressions) <= 15:
        for var, expr in expressions.items():
            print(f"  {var} = {expr}")
    else:
        print(f"  (共 {len(expressions)} 个表达式，省略显示)")
        # 只显示前几个
        for i, (var, expr) in enumerate(expressions.items()):
            if i >= 5:
                print(f"  ...")
                break
            print(f"  {var} = {expr}")
    
    # 运行测试
    results = run_test_case(test_case, verbose=True)
    
    # 保存结果
    results_file = os.path.join(test_dir, f"{test_name}_results.json")
    
    # 转换 set 为 list 以便 JSON 序列化
    serializable_results = {
        'num_expressions': results['num_expressions'],
        'num_changes': results['num_changes'],
        'has_cycles': results.get('has_cycles', False),
        'timing': results['timing'],
        'change_results': results.get('change_results', [])
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
