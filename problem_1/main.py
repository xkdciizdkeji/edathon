"""
Problem 1: 三值逻辑重替换 (Resubstitution in 3-Valued Logic)
主程序 - 整合电路生成、解析和重替换算法

使用方法：
1. 使用现有的测试用例文件：
   python main.py --bench test_cases/positive_simple.bench --gates test_cases/positive_simple_gates.txt

2. 生成并测试新的测试用例：
   python main.py --generate --num-inputs 3 --num-outputs 3 --positive

3. 运行所有内置测试：
   python main.py --test
"""

import argparse
import os
import sys
from typing import Optional

from truth_table_generator import generate_standard_gates, print_truth_table
from circuit_generator import (
    Circuit, 
    generate_positive_case, 
    generate_negative_case,
    generate_complex_positive_case,
    verify_resubstitution_exists,
    save_test_case
)
from bench_parser import BenchParser, load_circuit
from resubstitution_solver import ThreeValuedResubstitution, ResubstitutionResult


def run_on_files(bench_file: str, gates_file: str, verbose: bool = True) -> ResubstitutionResult:
    """
    对指定的电路文件运行重替换算法
    
    Args:
        bench_file: BENCH格式电路文件
        gates_file: 门真值表文件
        verbose: 是否打印详细信息
    
    Returns:
        重替换判定结果
    """
    if verbose:
        print(f"加载电路: {bench_file}")
        print(f"加载门定义: {gates_file}")
        print()
    
    # 解析电路
    parser = BenchParser()
    parser.parse_bench_file(bench_file)
    parser.parse_truth_table_file(gates_file)
    
    if verbose:
        parser.print_info()
        print()
    
    # 运行重替换算法
    solver = ThreeValuedResubstitution(parser)
    result = solver.solve()
    
    if verbose:
        print("=" * 60)
        print("重替换判定结果")
        print("=" * 60)
        print(f"最后输出: {parser.primary_outputs[-1]}")
        print(f"其他输出: {parser.primary_outputs[:-1]}")
        print()
        print(f"结论: {result}")
        print()
        
        if result.can_resubstitute:
            print("最后输出可以由其他输出表示！")
            solver.print_function_table(result)
        else:
            print("最后输出不能由其他输出唯一表示。")
            if result.conflict:
                print(f"\n反例（两个产生冲突的输入）:")
                print(f"  输入1: {result.conflict[0]}")
                print(f"  输入2: {result.conflict[1]}")
    
    return result


def generate_and_test(
    num_inputs: int = 3,
    num_outputs: int = 3,
    is_positive: bool = True,
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> ResubstitutionResult:
    """
    生成测试用例并运行重替换算法
    
    Args:
        num_inputs: 主输入数量
        num_outputs: 输出数量（包括最后一个）
        is_positive: True生成正例（可替换），False生成反例
        seed: 随机种子
        save_dir: 保存目录（None则不保存）
        verbose: 是否打印详细信息
    
    Returns:
        重替换判定结果
    """
    if verbose:
        case_type = "正例（可替换）" if is_positive else "反例（不可替换）"
        print(f"生成 {case_type}")
        print(f"参数: 输入数={num_inputs}, 输出数={num_outputs}, 种子={seed}")
        print()
    
    # 生成电路
    if is_positive:
        circuit, expected_func = generate_positive_case(
            num_primary_inputs=num_inputs,
            num_other_outputs=num_outputs - 1,
            seed=seed
        )
        if verbose:
            print(f"预期的替换函数: {expected_func}")
    else:
        circuit = generate_negative_case(
            num_primary_inputs=num_inputs,
            num_other_outputs=num_outputs - 1,
            seed=seed
        )
    
    if verbose:
        print("\n生成的电路（BENCH格式）:")
        print("-" * 40)
        print(circuit.to_bench_format())
        print("-" * 40)
        print()
    
    # 保存文件（如果指定）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = f"{'positive' if is_positive else 'negative'}_{num_inputs}in_{num_outputs}out"
        if seed is not None:
            base_name += f"_seed{seed}"
        
        circuit.save_to_bench(os.path.join(save_dir, f"{base_name}.bench"))
        circuit.save_gate_types(os.path.join(save_dir, f"{base_name}_gates.txt"))
        
        if verbose:
            print(f"已保存到 {save_dir}/{base_name}.*")
            print()
    
    # 首先用简单验证检查
    if verbose:
        print("使用简单枚举验证...")
    can_resub_simple, conflict_simple = verify_resubstitution_exists(circuit)
    
    if verbose:
        print(f"简单验证结果: {'可替换' if can_resub_simple else '不可替换'}")
        if conflict_simple:
            print(f"  冲突: {conflict_simple}")
        print()
    
    # 创建解析器并运行算法
    parser = BenchParser()
    parser.parse_bench_string(circuit.to_bench_format())
    
    # 添加门类型
    for gate_type, tt in circuit.gate_types.items():
        parser.gate_types[gate_type] = tt
    
    solver = ThreeValuedResubstitution(parser)
    result = solver.solve()
    
    if verbose:
        print("=" * 60)
        print("重替换算法结果")
        print("=" * 60)
        print(f"结论: {result}")
        
        if result.can_resubstitute:
            solver.print_function_table(result)
        elif result.conflict:
            print(f"\n反例输入对:")
            print(f"  输入1: {result.conflict[0]}")
            print(f"  输入2: {result.conflict[1]}")
    
    # 验证算法结果与简单验证一致
    if result.can_resubstitute != can_resub_simple:
        print("\n警告：算法结果与简单验证不一致！")
    
    return result


def run_builtin_tests():
    """运行内置的测试用例"""
    print("=" * 70)
    print("运行内置测试用例")
    print("=" * 70)
    print()
    
    test_results = []
    
    # 测试1：简单正例
    print("【测试1】简单正例 - 3输入2+1输出")
    print("-" * 70)
    result1 = generate_and_test(num_inputs=3, num_outputs=3, is_positive=True, seed=42)
    test_results.append(("简单正例", result1.can_resubstitute == True))
    print("\n")
    
    # 测试2：简单反例
    print("【测试2】简单反例 - 3输入2+1输出")
    print("-" * 70)
    result2 = generate_and_test(num_inputs=3, num_outputs=3, is_positive=False, seed=42)
    test_results.append(("简单反例", result2.can_resubstitute == False))
    print("\n")
    
    # 测试3：更多输入的正例
    print("【测试3】正例 - 4输入3+1输出")
    print("-" * 70)
    result3 = generate_and_test(num_inputs=4, num_outputs=4, is_positive=True, seed=123)
    test_results.append(("4输入正例", result3.can_resubstitute == True))
    print("\n")
    
    # 测试4：更多输入的反例
    print("【测试4】反例 - 4输入3+1输出")
    print("-" * 70)
    result4 = generate_and_test(num_inputs=4, num_outputs=4, is_positive=False, seed=123)
    test_results.append(("4输入反例", result4.can_resubstitute == False))
    print("\n")
    
    # 测试5：边界情况 - 只有2个输出
    print("【测试5】边界情况 - 2输入1+1输出")
    print("-" * 70)
    result5 = generate_and_test(num_inputs=2, num_outputs=2, is_positive=True, seed=789)
    test_results.append(("边界正例", result5.can_resubstitute == True))
    print("\n")
    
    # 测试6：题目中的经典例子（正确格式：所有输出都由主输入直接表示）
    # n1 = a + c (即 MAX(a, c))
    # n2 = b + c (即 MAX(b, c))  
    # n3 = ab + c (即 MAX(MIN(a,b), c))
    # 注意：n3 在逻辑上等于 n1 * n2 = MIN(n1, n2)，但电路中它由主输入直接表示
    print("【测试6】经典例子 - n1=MAX(a,c), n2=MAX(b,c), n3=MAX(MIN(a,b),c)")
    print("         （n3 逻辑上等于 MIN(n1, n2)，但电路中由主输入直接表示）")
    print("-" * 70)
    
    bench_content = """
INPUT(a)
INPUT(b)
INPUT(c)
OUTPUT(n1)
OUTPUT(n2)
OUTPUT(n3)
n1 = MAX(a, c)
n2 = MAX(b, c)
_temp_ab = MIN(a, b)
n3 = MAX(_temp_ab, c)
"""
    
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
    
    solver = ThreeValuedResubstitution(parser)
    result6 = solver.solve()
    
    print(f"结论: {result6}")
    if result6.can_resubstitute:
        solver.print_function_table(result6)
    test_results.append(("经典例子", result6.can_resubstitute == True))
    print("\n")
    
    # 总结
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    
    all_passed = True
    for name, passed in test_results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("所有测试通过！")
    else:
        print("存在失败的测试。")
    
    return all_passed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="三值逻辑重替换算法 (Problem 1: Resubstitution in 3-Valued Logic)"
    )
    
    parser.add_argument(
        "--bench", "-b",
        help="BENCH格式电路文件路径"
    )
    parser.add_argument(
        "--gates", "-g",
        help="门真值表文件路径"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="生成新的测试用例"
    )
    parser.add_argument(
        "--positive",
        action="store_true",
        help="生成正例（可替换）"
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="生成反例（不可替换）"
    )
    parser.add_argument(
        "--num-inputs", "-i",
        type=int,
        default=3,
        help="主输入数量（默认：3）"
    )
    parser.add_argument(
        "--num-outputs", "-o",
        type=int,
        default=3,
        help="输出数量（默认：3）"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="随机种子"
    )
    parser.add_argument(
        "--save-dir",
        help="保存生成的测试用例到指定目录"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="运行内置测试"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（只输出最终结果）"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.test:
        # 运行内置测试
        success = run_builtin_tests()
        sys.exit(0 if success else 1)
    
    elif args.bench and args.gates:
        # 使用指定的文件
        result = run_on_files(args.bench, args.gates, verbose=verbose)
        
        # 输出简洁结果（供自动化使用）
        if args.quiet:
            print("YES" if result.can_resubstitute else "NO")
    
    elif args.generate:
        # 生成测试用例
        is_positive = args.positive or not args.negative  # 默认生成正例
        
        result = generate_and_test(
            num_inputs=args.num_inputs,
            num_outputs=args.num_outputs,
            is_positive=is_positive,
            seed=args.seed,
            save_dir=args.save_dir,
            verbose=verbose
        )
        
        if args.quiet:
            print("YES" if result.can_resubstitute else "NO")
    
    else:
        # 没有参数，运行简单演示
        print("三值逻辑重替换算法 - 演示模式")
        print("使用 --help 查看所有选项")
        print()
        print("运行内置测试...")
        print()
        run_builtin_tests()


if __name__ == "__main__":
    main()
