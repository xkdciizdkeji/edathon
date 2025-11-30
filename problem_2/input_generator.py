"""
微流控芯片布线问题 - 输入数据生成器
Microfluidic Chip Routing Problem - Input Generator

根据 ISO 22916:2022 标准，生成微流控芯片的测试用例，包括：
- 芯片板尺寸
- 模块（obstacles）位置和尺寸
- 需要布线的通道（nets）的源端口和目标端口
- 约束参数（最小间距、最大waypoints数等）

作者: EDAthon 2025 参赛预备代码
"""

import json
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Port:
    """端口类 - 表示模块上的连接点"""
    id: str           # 端口唯一标识符
    x: float          # x坐标
    y: float          # y坐标
    module_id: str    # 所属模块ID（如果是芯片边缘端口则为 "board"）


@dataclass
class Module:
    """模块类 - 表示芯片上的功能模块（障碍物）"""
    id: str           # 模块唯一标识符
    x: float          # 左下角x坐标
    y: float          # 左下角y坐标
    width: float      # 宽度
    height: float     # 高度
    ports: List[Port] # 模块上的端口列表


@dataclass
class Net:
    """网络类 - 表示需要连接的一对端口"""
    id: str           # 网络唯一标识符
    source_port: str  # 源端口ID
    target_port: str  # 目标端口ID


@dataclass
class ChipSpec:
    """芯片规格类 - 包含所有设计约束"""
    board_width: float        # 芯片板宽度
    board_height: float       # 芯片板高度
    min_spacing: float        # 通道间最小间距
    max_waypoints: int        # 每条通道最大waypoints数
    channel_width: float      # 通道宽度


@dataclass
class MicrofluidicChip:
    """微流控芯片类 - 完整的测试用例"""
    spec: ChipSpec
    modules: List[Module]
    board_ports: List[Port]   # 芯片边缘的端口
    nets: List[Net]


def generate_random_modules(
    board_width: float,
    board_height: float,
    num_modules: int,
    min_module_size: float = 50,
    max_module_size: float = 150,
    min_spacing: float = 30,
    ports_per_module: Tuple[int, int] = (2, 4)
) -> List[Module]:
    """
    在芯片板上随机生成不重叠的模块
    
    算法：
    1. 随机生成模块位置和尺寸
    2. 检查与已有模块的间距约束
    3. 在模块边缘随机放置端口
    
    Args:
        board_width: 芯片板宽度
        board_height: 芯片板高度
        num_modules: 要生成的模块数量
        min_module_size: 模块最小尺寸
        max_module_size: 模块最大尺寸
        min_spacing: 模块间最小间距
        ports_per_module: 每个模块端口数量范围 (min, max)
    
    Returns:
        生成的模块列表
    """
    modules = []
    max_attempts = 1000  # 防止无限循环
    
    for i in range(num_modules):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # 随机生成模块尺寸
            width = random.uniform(min_module_size, max_module_size)
            height = random.uniform(min_module_size, max_module_size)
            
            # 随机生成位置（确保在板内）
            margin = min_spacing * 2
            x = random.uniform(margin, board_width - width - margin)
            y = random.uniform(margin, board_height - height - margin)
            
            # 检查与已有模块的间距
            valid = True
            for existing in modules:
                # 计算两个模块之间的最小距离
                dx = max(0, max(existing.x - (x + width), x - (existing.x + existing.width)))
                dy = max(0, max(existing.y - (y + height), y - (existing.y + existing.height)))
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < min_spacing:
                    valid = False
                    break
            
            if valid:
                # 创建模块
                module_id = f"M{i}"
                ports = generate_module_ports(
                    module_id, x, y, width, height, 
                    random.randint(ports_per_module[0], ports_per_module[1])
                )
                
                module = Module(
                    id=module_id,
                    x=x, y=y,
                    width=width, height=height,
                    ports=ports
                )
                modules.append(module)
                placed = True
        
        if not placed:
            print(f"Warning: Could not place module {i} after {max_attempts} attempts")
    
    return modules


def generate_module_ports(
    module_id: str,
    x: float, y: float,
    width: float, height: float,
    num_ports: int
) -> List[Port]:
    """
    在模块边缘生成端口
    
    端口均匀分布在模块的四条边上
    
    Args:
        module_id: 模块ID
        x, y: 模块左下角坐标
        width, height: 模块尺寸
        num_ports: 端口数量
    
    Returns:
        端口列表
    """
    ports = []
    edges = ['top', 'bottom', 'left', 'right']
    
    for i in range(num_ports):
        port_id = f"{module_id}_P{i}"
        edge = edges[i % 4]
        
        # 在选定边上随机选择位置
        offset_ratio = random.uniform(0.2, 0.8)
        
        if edge == 'top':
            px = x + width * offset_ratio
            py = y + height
        elif edge == 'bottom':
            px = x + width * offset_ratio
            py = y
        elif edge == 'left':
            px = x
            py = y + height * offset_ratio
        else:  # right
            px = x + width
            py = y + height * offset_ratio
        
        ports.append(Port(id=port_id, x=px, y=py, module_id=module_id))
    
    return ports


def generate_board_ports(
    board_width: float,
    board_height: float,
    num_ports: int,
    margin: float = 20
) -> List[Port]:
    """
    在芯片板边缘生成端口（外部接口）
    
    Args:
        board_width: 芯片板宽度
        board_height: 芯片板高度
        num_ports: 端口数量
        margin: 边缘留白
    
    Returns:
        端口列表
    """
    ports = []
    edges = ['top', 'bottom', 'left', 'right']
    ports_per_edge = num_ports // 4
    
    for edge_idx, edge in enumerate(edges):
        for i in range(ports_per_edge):
            port_id = f"B_{edge}_{i}"
            
            # 在边上均匀分布
            ratio = (i + 1) / (ports_per_edge + 1)
            
            if edge == 'top':
                px = margin + (board_width - 2 * margin) * ratio
                py = board_height
            elif edge == 'bottom':
                px = margin + (board_width - 2 * margin) * ratio
                py = 0
            elif edge == 'left':
                px = 0
                py = margin + (board_height - 2 * margin) * ratio
            else:  # right
                px = board_width
                py = margin + (board_height - 2 * margin) * ratio
            
            ports.append(Port(id=port_id, x=px, y=py, module_id="board"))
    
    return ports


def generate_nets(
    modules: List[Module],
    board_ports: List[Port],
    num_nets: int,
    connect_to_board_ratio: float = 0.3
) -> List[Net]:
    """
    生成需要布线的网络（端口对）
    
    策略：
    - 部分网络连接模块端口到芯片边缘端口
    - 部分网络连接不同模块的端口
    
    Args:
        modules: 模块列表
        board_ports: 芯片边缘端口列表
        num_nets: 要生成的网络数量
        connect_to_board_ratio: 连接到边缘端口的比例
    
    Returns:
        网络列表
    """
    nets = []
    
    # 收集所有模块端口
    module_ports = []
    for module in modules:
        module_ports.extend(module.ports)
    
    if not module_ports:
        return nets
    
    used_ports = set()  # 跟踪已使用的端口
    
    for i in range(num_nets):
        net_id = f"N{i}"
        
        # 随机选择源端口（从模块端口中选择）
        available_module_ports = [p for p in module_ports if p.id not in used_ports]
        if not available_module_ports:
            break
        
        source = random.choice(available_module_ports)
        used_ports.add(source.id)
        
        # 决定目标端口类型
        if random.random() < connect_to_board_ratio and board_ports:
            # 连接到芯片边缘
            available_board_ports = [p for p in board_ports if p.id not in used_ports]
            if available_board_ports:
                target = random.choice(available_board_ports)
                used_ports.add(target.id)
            else:
                continue
        else:
            # 连接到另一个模块端口
            available_targets = [
                p for p in module_ports 
                if p.id not in used_ports and p.module_id != source.module_id
            ]
            if available_targets:
                target = random.choice(available_targets)
                used_ports.add(target.id)
            else:
                continue
        
        nets.append(Net(id=net_id, source_port=source.id, target_port=target.id))
    
    return nets


def generate_microfluidic_chip(
    board_width: float = 1000,
    board_height: float = 800,
    num_modules: int = 5,
    num_board_ports: int = 8,
    num_nets: int = 10,
    min_spacing: float = 20,
    max_waypoints: int = 6,
    channel_width: float = 5,
    seed: Optional[int] = None
) -> MicrofluidicChip:
    """
    生成完整的微流控芯片测试用例
    
    Args:
        board_width: 芯片板宽度
        board_height: 芯片板高度
        num_modules: 模块数量
        num_board_ports: 边缘端口数量
        num_nets: 网络数量
        min_spacing: 最小间距
        max_waypoints: 最大waypoints数
        channel_width: 通道宽度
        seed: 随机种子（用于可重复生成）
    
    Returns:
        完整的微流控芯片对象
    """
    if seed is not None:
        random.seed(seed)
    
    # 创建规格
    spec = ChipSpec(
        board_width=board_width,
        board_height=board_height,
        min_spacing=min_spacing,
        max_waypoints=max_waypoints,
        channel_width=channel_width
    )
    
    # 生成模块
    modules = generate_random_modules(
        board_width, board_height, num_modules,
        min_spacing=min_spacing * 2
    )
    
    # 生成边缘端口
    board_ports = generate_board_ports(board_width, board_height, num_board_ports)
    
    # 生成网络
    nets = generate_nets(modules, board_ports, num_nets)
    
    return MicrofluidicChip(
        spec=spec,
        modules=modules,
        board_ports=board_ports,
        nets=nets
    )


def chip_to_dict(chip: MicrofluidicChip) -> dict:
    """将芯片对象转换为可JSON序列化的字典"""
    return {
        "spec": asdict(chip.spec),
        "modules": [
            {
                "id": m.id,
                "x": m.x, "y": m.y,
                "width": m.width, "height": m.height,
                "ports": [asdict(p) for p in m.ports]
            }
            for m in chip.modules
        ],
        "board_ports": [asdict(p) for p in chip.board_ports],
        "nets": [asdict(n) for n in chip.nets]
    }


def save_chip_to_json(chip: MicrofluidicChip, filename: str):
    """将芯片保存为JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chip_to_dict(chip), f, indent=2, ensure_ascii=False)
    print(f"Saved chip to {filename}")


def load_chip_from_json(filename: str) -> MicrofluidicChip:
    """从JSON文件加载芯片"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    spec = ChipSpec(**data['spec'])
    
    modules = []
    for m in data['modules']:
        ports = [Port(**p) for p in m['ports']]
        modules.append(Module(
            id=m['id'], x=m['x'], y=m['y'],
            width=m['width'], height=m['height'],
            ports=ports
        ))
    
    board_ports = [Port(**p) for p in data['board_ports']]
    nets = [Net(**n) for n in data['nets']]
    
    return MicrofluidicChip(spec=spec, modules=modules, board_ports=board_ports, nets=nets)


def generate_test_cases():
    """生成一系列测试用例"""
    test_cases = [
        # 简单测试：少量模块和网络
        {
            "name": "simple",
            "params": {
                "board_width": 500, "board_height": 400,
                "num_modules": 3, "num_board_ports": 4, "num_nets": 4,
                "min_spacing": 15, "max_waypoints": 8, "seed": 42
            }
        },
        # 中等测试
        {
            "name": "medium",
            "params": {
                "board_width": 800, "board_height": 600,
                "num_modules": 5, "num_board_ports": 8, "num_nets": 8,
                "min_spacing": 20, "max_waypoints": 10, "seed": 123
            }
        },
        # 复杂测试：更多模块和网络
        {
            "name": "complex",
            "params": {
                "board_width": 1200, "board_height": 900,
                "num_modules": 8, "num_board_ports": 12, "num_nets": 15,
                "min_spacing": 25, "max_waypoints": 12, "seed": 456
            }
        },
        # 困难测试：紧凑布局
        {
            "name": "hard",
            "params": {
                "board_width": 600, "board_height": 500,
                "num_modules": 6, "num_board_ports": 8, "num_nets": 12,
                "min_spacing": 15, "max_waypoints": 10, "seed": 789
            }
        },
        # 大规模测试
        {
            "name": "large",
            "params": {
                "board_width": 1500, "board_height": 1200,
                "num_modules": 12, "num_board_ports": 16, "num_nets": 20,
                "min_spacing": 30, "max_waypoints": 15, "seed": 999
            }
        }
    ]
    
    return test_cases


if __name__ == "__main__":
    import os
    
    # 创建测试用例目录
    test_dir = os.path.join(os.path.dirname(__file__), "test_cases")
    os.makedirs(test_dir, exist_ok=True)
    
    # 生成所有测试用例
    test_cases = generate_test_cases()
    
    for tc in test_cases:
        chip = generate_microfluidic_chip(**tc["params"])
        filename = os.path.join(test_dir, f"{tc['name']}.json")
        save_chip_to_json(chip, filename)
        
        # 打印统计信息
        print(f"\n=== Test Case: {tc['name']} ===")
        print(f"  Board: {chip.spec.board_width} x {chip.spec.board_height}")
        print(f"  Modules: {len(chip.modules)}")
        print(f"  Board Ports: {len(chip.board_ports)}")
        print(f"  Nets: {len(chip.nets)}")
        print(f"  Min Spacing: {chip.spec.min_spacing}")
        print(f"  Max Waypoints: {chip.spec.max_waypoints}")
    
    print("\n=== All test cases generated! ===")
