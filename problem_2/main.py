"""
微流控芯片布线问题 - 主程序入口
Microfluidic Chip Routing Problem - Main Program

本程序是 EDAthon 2025 Problem 2 的完整解决方案。
可以用于：
1. 生成测试用例
2. 求解布线问题
3. 可视化结果
4. 输出统计信息

使用方法:
    # 生成测试用例
    python main.py --generate
    
    # 求解指定测试用例
    python main.py --solve test_cases/simple.json
    
    # 求解并可视化
    python main.py --solve test_cases/simple.json --visualize
    
    # 运行所有测试用例
    python main.py --run-all

作者: EDAthon 2025 参赛预备代码
"""

import argparse
import os
import sys
import time
import json
from typing import Dict, List

# 导入本地模块
from input_generator import (
    MicrofluidicChip, generate_microfluidic_chip, 
    save_chip_to_json, load_chip_from_json, generate_test_cases
)
from router import solve_microfluidic_routing, Route, MicrofluidicRouter


def generate_all_test_cases(output_dir: str):
    """
    生成所有预定义的测试用例
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test_cases = generate_test_cases()
    
    print("=" * 60)
    print("Generating Test Cases for Microfluidic Routing")
    print("=" * 60)
    
    for tc in test_cases:
        chip = generate_microfluidic_chip(**tc["params"])
        filename = os.path.join(output_dir, f"{tc['name']}.json")
        save_chip_to_json(chip, filename)
        
        print(f"\n[{tc['name']}]")
        print(f"  Board: {chip.spec.board_width} x {chip.spec.board_height}")
        print(f"  Modules: {len(chip.modules)}")
        print(f"  Nets: {len(chip.nets)}")
        print(f"  Min Spacing: {chip.spec.min_spacing}")
        print(f"  Max Waypoints: {chip.spec.max_waypoints}")
        print(f"  Saved to: {filename}")
    
    print("\n" + "=" * 60)
    print(f"Generated {len(test_cases)} test cases in '{output_dir}'")
    print("=" * 60)


def solve_single_case(
    input_file: str, 
    visualize: bool = False,
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    求解单个测试用例
    
    Args:
        input_file: 输入JSON文件路径
        visualize: 是否可视化结果
        output_dir: 输出目录（用于保存结果）
        verbose: 是否打印详细信息
    
    Returns:
        包含结果统计的字典
    """
    if verbose:
        print(f"\nLoading: {input_file}")
    
    # 加载芯片
    chip = load_chip_from_json(input_file)
    
    if verbose:
        print(f"  Board: {chip.spec.board_width} x {chip.spec.board_height}")
        print(f"  Modules: {len(chip.modules)}")
        print(f"  Nets: {len(chip.nets)}")
        print(f"  Min Spacing: {chip.spec.min_spacing}")
        print(f"  Max Waypoints: {chip.spec.max_waypoints}")
    
    # 布线
    if verbose:
        print("\nRouting...")
    
    start_time = time.time()
    routes, stats = solve_microfluidic_routing(chip, use_rip_up=True)
    elapsed_time = time.time() - start_time
    
    stats['runtime_seconds'] = elapsed_time
    
    if verbose:
        print(f"\n{'=' * 40}")
        print("ROUTING RESULTS")
        print(f"{'=' * 40}")
        print(f"  Routed:       {stats['routed_nets']} / {stats['total_nets']}")
        print(f"  Success Rate: {stats['routed_nets']/stats['total_nets']*100:.1f}%")
        print(f"  Failed:       {stats['failed_nets']}")
        print(f"  Total Length: {stats['total_length']:.2f}")
        print(f"  Total Turns:  {stats['total_turns']}")
        print(f"  Runtime:      {elapsed_time:.3f} seconds")
        print(f"{'=' * 40}")
        
        # 详细路由信息
        print("\nRoute Details:")
        for net_id, route in routes.items():
            print(f"  {net_id}: {len(route.waypoints)} waypoints, "
                  f"length={route.total_length():.1f}, turns={route.num_turns()}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取基本文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 保存路由结果
        result_file = os.path.join(output_dir, f"{base_name}_result.json")
        save_routing_result(routes, stats, result_file)
        
        if verbose:
            print(f"\nSaved result to: {result_file}")
    
    # 可视化
    if visualize:
        try:
            from visualizer import visualize_chip
            import matplotlib.pyplot as plt
            
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            if output_dir:
                save_path = os.path.join(output_dir, f"{base_name}_visualization.png")
            else:
                save_path = None
            
            fig = visualize_chip(
                chip, routes,
                title=f"Microfluidic Routing: {base_name}",
                show_grid=True,
                save_path=save_path
            )
            
            plt.show()
            
        except ImportError as e:
            print(f"Warning: Could not visualize. {e}")
    
    return stats


def save_routing_result(routes: Dict[str, Route], stats: dict, filename: str):
    """
    保存布线结果到JSON文件
    
    Args:
        routes: 布线结果
        stats: 统计信息
        filename: 输出文件名
    """
    result = {
        "statistics": stats,
        "routes": {}
    }
    
    for net_id, route in routes.items():
        result["routes"][net_id] = {
            "waypoints": [{"x": p.x, "y": p.y} for p in route.waypoints],
            "num_waypoints": len(route.waypoints),
            "total_length": route.total_length(),
            "num_turns": route.num_turns()
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_all_test_cases(test_dir: str, output_dir: str = None):
    """
    运行所有测试用例并汇总结果
    
    Args:
        test_dir: 测试用例目录
        output_dir: 输出目录
    """
    print("=" * 70)
    print("Running All Test Cases")
    print("=" * 70)
    
    # 查找所有JSON测试文件
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    test_files.sort()
    
    if not test_files:
        print(f"No test cases found in '{test_dir}'")
        return
    
    results = []
    
    for test_file in test_files:
        input_path = os.path.join(test_dir, test_file)
        
        print(f"\n{'─' * 50}")
        print(f"Test Case: {test_file}")
        print(f"{'─' * 50}")
        
        stats = solve_single_case(
            input_path, 
            visualize=False, 
            output_dir=output_dir,
            verbose=True
        )
        
        results.append({
            "test_case": test_file,
            **stats
        })
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Test Case':<30} {'Routed':<12} {'Success%':<12} {'Runtime':<12}")
    print("-" * 70)
    
    total_routed = 0
    total_nets = 0
    total_runtime = 0
    
    for r in results:
        success_rate = r['routed_nets'] / r['total_nets'] * 100 if r['total_nets'] > 0 else 0
        print(f"{r['test_case']:<30} "
              f"{r['routed_nets']}/{r['total_nets']:<8} "
              f"{success_rate:>6.1f}%      "
              f"{r['runtime_seconds']:.3f}s")
        
        total_routed += r['routed_nets']
        total_nets += r['total_nets']
        total_runtime += r['runtime_seconds']
    
    print("-" * 70)
    overall_rate = total_routed / total_nets * 100 if total_nets > 0 else 0
    print(f"{'TOTAL':<30} {total_routed}/{total_nets:<8} {overall_rate:>6.1f}%      {total_runtime:.3f}s")
    print("=" * 70)
    
    # 保存汇总结果
    if output_dir:
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "summary": {
                    "total_routed": total_routed,
                    "total_nets": total_nets,
                    "overall_success_rate": overall_rate,
                    "total_runtime": total_runtime
                }
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="Microfluidic Chip Routing Solver - EDAthon 2025 Problem 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate test cases
  python main.py --generate
  
  # Solve a single test case
  python main.py --solve test_cases/simple.json
  
  # Solve with visualization
  python main.py --solve test_cases/medium.json --visualize
  
  # Run all test cases
  python main.py --run-all
  
  # Custom test case generation
  python main.py --generate --output custom_tests
        """
    )
    
    parser.add_argument(
        '--generate', 
        action='store_true',
        help='Generate test cases'
    )
    
    parser.add_argument(
        '--solve',
        type=str,
        metavar='FILE',
        help='Solve a specific test case (JSON file)'
    )
    
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all test cases in the test_cases directory'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize the routing result'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='test_cases',
        help='Directory containing test cases (default: test_cases)'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    test_dir = os.path.join(script_dir, args.test_dir)
    output_dir = os.path.join(script_dir, args.output)
    
    # 执行操作
    if args.generate:
        generate_all_test_cases(test_dir)
    
    elif args.solve:
        # 处理输入文件路径
        if not os.path.isabs(args.solve):
            input_file = os.path.join(script_dir, args.solve)
        else:
            input_file = args.solve
        
        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)
        
        solve_single_case(
            input_file, 
            visualize=args.visualize,
            output_dir=output_dir
        )
    
    elif args.run_all:
        if not os.path.exists(test_dir):
            print(f"Error: Test directory not found: {test_dir}")
            print("Please run with --generate first to create test cases.")
            sys.exit(1)
        
        run_all_test_cases(test_dir, output_dir)
    
    else:
        # 默认行为：显示帮助或运行一个简单示例
        print("Microfluidic Chip Routing Solver")
        print("-" * 40)
        print("\nNo action specified. Running a quick demo...\n")
        
        # 生成一个简单的测试用例并求解
        chip = generate_microfluidic_chip(
            board_width=500,
            board_height=400,
            num_modules=3,
            num_board_ports=4,
            num_nets=5,
            min_spacing=15,
            max_waypoints=6,
            seed=42
        )
        
        print(f"Generated demo chip:")
        print(f"  Board: {chip.spec.board_width} x {chip.spec.board_height}")
        print(f"  Modules: {len(chip.modules)}")
        print(f"  Nets: {len(chip.nets)}")
        
        routes, stats = solve_microfluidic_routing(chip)
        
        print(f"\nRouting Results:")
        print(f"  Routed: {stats['routed_nets']} / {stats['total_nets']}")
        print(f"  Success Rate: {stats['routed_nets']/stats['total_nets']*100:.1f}%")
        
        print("\nUse --help for more options.")


if __name__ == "__main__":
    main()



'''
# 运行方式示例:
# # 生成测试用例
python main.py --generate

# 求解单个测试用例
python main.py --solve test_cases/simple.json

# 求解并可视化
python main.py --solve test_cases/simple.json --visualize

# 运行所有测试用例
python main.py --run-all



核心模块说明
1. input_generator.py - 输入数据生成器
ChipSpec: 芯片规格（尺寸、最小间距、waypoints限制等）
Module: 模块定义（位置、尺寸、端口）
Port: 端口定义（坐标、所属模块）
Net: 网络定义（源端口、目标端口）
generate_microfluidic_chip(): 生成完整测试用例
2. router.py - 布线求解器（核心算法）
算法: 带转弯限制的 A* 搜索
关键特性:
网格化空间（cell_size = min_spacing / 2）
模块障碍物处理 + 端口通道清理
转弯惩罚机制（优先直线路径）
Rip-up & Reroute 冲突处理策略
路径简化（移除冗余waypoints）




算法核心注释

A 搜索关键点*:
# 状态空间: (位置, 方向, 转弯次数, 路径)
# 代价函数: g = 移动距离 + 转弯惩罚
# 启发函数: h = 曼哈顿距离到终点
# 约束: turns <= max_waypoints - 2

Rip-up & Reroute 策略:
对失败的网络，找到可能冲突的已布线
移除部分冲突路由，释放空间
打乱优先级重新布线
多次迭代直到收敛
'''