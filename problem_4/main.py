"""
主程序入口 (Main Entry Point)

该程序用于测试和验证Dominant Device识别算法。

功能:
1. 生成测试用例
2. 运行Dominant Device识别算法
3. 验证算法结果与预期结果的一致性
4. 生成测试报告

使用方法:
    python main.py                    # 运行所有测试
    python main.py --generate         # 仅生成测试用例
    python main.py --test <name>      # 运行指定测试用例
    python main.py --verbose          # 详细输出模式
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Set, Tuple

from circuit_parser import parse_spice_netlist, print_circuit_info
from test_generator import generate_all_test_cases, save_test_cases, TestCase
from dominant_finder import DominantDeviceFinder, find_dominant_devices


def run_single_test(test_case: TestCase, verbose: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    运行单个测试用例
    
    参数:
        test_case: 测试用例对象
        verbose: 是否详细输出
        
    返回:
        Tuple[bool, List[str], List[str]]: (是否通过, 识别结果, 预期结果)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"测试用例: {test_case.name}")
        print(f"描述: {test_case.description}")
        print(f"难度: {test_case.difficulty}")
        print(f"输入跳变: {test_case.input_transitions}")
        print(f"预期dominant: {test_case.expected_dominant}")
        print(f"{'='*60}")
    
    # 运行算法
    identified = find_dominant_devices(
        test_case.netlist,
        test_case.input_transitions,
        verbose=verbose
    )
    
    # 验证结果
    expected_set = set(test_case.expected_dominant)
    identified_set = set(identified)
    
    # 检查是否所有预期的dominant devices都被识别出来
    missing = expected_set - identified_set
    extra = identified_set - expected_set
    
    # 通过条件: 识别出了所有预期的dominant devices
    # 允许识别出额外的器件(可能也是重要的)
    passed = len(missing) == 0
    
    if verbose:
        print(f"\n[验证结果]")
        print(f"  识别的dominant: {identified}")
        print(f"  预期的dominant: {test_case.expected_dominant}")
        
        if missing:
            print(f"  缺失: {list(missing)}")
        if extra:
            print(f"  额外识别: {list(extra)}")
        
        if passed:
            print(f"\n  ✓ 测试通过")
        else:
            print(f"\n  ✗ 测试失败 (缺失: {list(missing)})")
    
    return passed, identified, test_case.expected_dominant


def run_all_tests(verbose: bool = False) -> Dict:
    """
    运行所有测试用例
    
    参数:
        verbose: 是否详细输出
        
    返回:
        Dict: 测试结果汇总
    """
    test_cases = generate_all_test_cases()
    
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    print("\n" + "="*70)
    print("运行 Dominant Device 识别算法测试")
    print("="*70)
    
    for tc in test_cases:
        passed, identified, expected = run_single_test(tc, verbose=verbose)
        
        result_detail = {
            "name": tc.name,
            "difficulty": tc.difficulty,
            "passed": passed,
            "identified": identified,
            "expected": expected
        }
        results["details"].append(result_detail)
        
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # 打印汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"总计: {results['total']} 个测试用例")
    print(f"通过: {results['passed']} 个")
    print(f"失败: {results['failed']} 个")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")
    
    # 按难度统计
    difficulty_stats = {}
    for detail in results["details"]:
        diff = detail["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "passed": 0}
        difficulty_stats[diff]["total"] += 1
        if detail["passed"]:
            difficulty_stats[diff]["passed"] += 1
    
    print(f"\n按难度统计:")
    for diff, stats in sorted(difficulty_stats.items()):
        rate = stats["passed"] / stats["total"] * 100
        print(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    # 列出失败的测试
    if results["failed"] > 0:
        print(f"\n失败的测试用例:")
        for detail in results["details"]:
            if not detail["passed"]:
                print(f"  - {detail['name']}")
                print(f"    预期: {detail['expected']}")
                print(f"    识别: {detail['identified']}")
    
    return results


def demo_two_stage_inverter():
    """
    演示: 两级反相器分析
    
    这是题目中给出的示例，用于验证算法的基本正确性。
    """
    print("\n" + "="*70)
    print("演示: 两级反相器 (题目示例)")
    print("="*70)
    
    netlist = """
    * Two-stage Inverter (两级反相器)
    * 电路结构: IN -> [INV1] -> MID -> [INV2] -> OUT
    
    .subckt INV2 OUT IN VDD GND
    .inputs IN
    .outputs OUT
    
    * 第一级反相器
    * MN1: NMOS, 当IN=1时导通, 拉低MID
    * MP1: PMOS, 当IN=0时导通, 拉高MID
    MN1 MID IN GND GND nmos W=1u L=0.1u
    MP1 MID IN VDD VDD pmos W=2u L=0.1u
    
    * 第二级反相器
    * MN2: NMOS, 当MID=1时导通, 拉低OUT
    * MP2: PMOS, 当MID=0时导通, 拉高OUT
    MN2 OUT MID GND GND nmos W=1u L=0.1u
    MP2 OUT MID VDD VDD pmos W=2u L=0.1u
    
    .ends INV2
    """
    
    print("\n[场景1] 输入 0→1 (上升沿)")
    print("-" * 40)
    print("分析:")
    print("  - 输入IN从0变为1")
    print("  - 第一级: MN1导通(gate=1), 拉低MID (MID: 1→0)")
    print("           MP1截止(gate=1)")
    print("  - 第二级: MP2导通(gate=0), 拉高OUT (OUT: 0→1)")
    print("           MN2截止(gate=0)")
    print("  预期dominant: MN1, MP2")
    
    dominant = find_dominant_devices(
        netlist,
        input_transitions={"IN": "rise"},
        verbose=True
    )
    
    print("\n" + "-" * 40)
    print("\n[场景2] 输入 1→0 (下降沿)")
    print("-" * 40)
    print("分析:")
    print("  - 输入IN从1变为0")
    print("  - 第一级: MP1导通(gate=0), 拉高MID (MID: 0→1)")
    print("           MN1截止(gate=0)")
    print("  - 第二级: MN2导通(gate=1), 拉低OUT (OUT: 1→0)")
    print("           MP2截止(gate=1)")
    print("  预期dominant: MP1, MN2")
    
    dominant = find_dominant_devices(
        netlist,
        input_transitions={"IN": "fall"},
        verbose=True
    )


def interactive_mode():
    """
    交互模式: 允许用户输入自定义电路
    """
    print("\n" + "="*70)
    print("交互模式: 自定义电路分析")
    print("="*70)
    print("\n请输入SPICE网表 (输入空行结束):")
    
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    
    if not lines:
        print("未输入任何内容")
        return
    
    netlist = "\n".join(lines)
    
    # 解析电路
    circuit = parse_spice_netlist(netlist)
    print_circuit_info(circuit)
    
    # 获取输入跳变
    print("\n请输入跳变信息 (格式: 输入名=rise/fall, 用空格分隔):")
    print(f"可用输入: {circuit.inputs}")
    
    transition_str = input().strip()
    input_transitions = {}
    
    for part in transition_str.split():
        if "=" in part:
            name, trans = part.split("=")
            input_transitions[name.strip()] = trans.strip()
    
    if not input_transitions:
        print("未指定输入跳变")
        return
    
    # 运行分析
    dominant = find_dominant_devices(
        netlist,
        input_transitions,
        verbose=True
    )
    
    print(f"\n识别的Dominant Devices: {dominant}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Dominant Device 识别算法测试程序"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="生成测试用例文件"
    )
    parser.add_argument(
        "--test", "-t",
        type=str,
        default=None,
        help="运行指定的测试用例 (名称)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出模式"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="运行演示 (两级反相器)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="运行所有测试"
    )
    
    args = parser.parse_args()
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "test_cases")
    
    if args.generate:
        # 生成测试用例
        test_cases = generate_all_test_cases()
        save_test_cases(test_cases, test_dir)
        print(f"测试用例已保存到: {test_dir}")
        
    elif args.demo:
        # 运行演示
        demo_two_stage_inverter()
        
    elif args.interactive:
        # 交互模式
        interactive_mode()
        
    elif args.test:
        # 运行指定测试
        test_cases = generate_all_test_cases()
        found = False
        for tc in test_cases:
            if tc.name == args.test:
                run_single_test(tc, verbose=True)
                found = True
                break
        if not found:
            print(f"未找到测试用例: {args.test}")
            print("可用的测试用例:")
            for tc in test_cases:
                print(f"  - {tc.name}")
    
    elif args.all or len(sys.argv) == 1:
        # 默认: 运行所有测试
        run_all_tests(verbose=args.verbose)
        
        # 同时运行演示
        demo_two_stage_inverter()


if __name__ == "__main__":
    main()

'''
# 运行方式示例:
python main.py --demo        # 运行题目示例演示
python main.py --all         # 运行所有测试
python main.py --verbose     # 详细输出模式
python main.py --generate    # 生成测试用例文件
python main.py --interactive # 交互模式（自定义电路）

算法核心思想
电路解析 (circuit_parser.py):

解析SPICE网表格式的电路描述
提取MOS管信息（NMOS/PMOS、端口连接、尺寸等）
建立节点连接关系图
信号传播分析 (dominant_finder.py):

根据输入跳变（rise/fall）确定输入节点状态
迭代传播信号到内部节点
判断每个MOS管的导通状态：
NMOS: gate为高电平时导通（放电路径）
PMOS: gate为低电平时导通（充电路径）
敏感性评分:

导通状态分 (50分): 非导通器件直接排除
关键路径分 (30分): 在信号传播路径上的器件
位置分 (15-10分): 直接驱动输出或第一级器件
类型匹配分 (20分): 输出上升时PMOS更重要，下降时NMOS更重要
尺寸因素 (0-10分): W/L比值影响
结果筛选: 按敏感性得分排序，超过阈值的器件为dominant devices
'''
