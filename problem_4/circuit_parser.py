"""
电路网表解析器 (Circuit Netlist Parser)

该模块用于解析SPICE格式的电路网表，提取MOS管信息和电路拓扑结构。
支持解析NMOS和PMOS器件，并建立节点连接关系。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum


class MOSType(Enum):
    """MOS管类型枚举"""
    NMOS = "nmos"
    PMOS = "pmos"


@dataclass
class MOSDevice:
    """
    MOS晶体管数据结构
    
    属性:
        name: 器件名称 (如 MN1, MP1)
        mos_type: MOS类型 (NMOS或PMOS)
        drain: 漏极连接节点
        gate: 栅极连接节点
        source: 源极连接节点
        bulk: 衬底连接节点
        width: 沟道宽度 (单位: um)
        length: 沟道长度 (单位: um)
        model: 模型名称
    """
    name: str
    mos_type: MOSType
    drain: str
    gate: str
    source: str
    bulk: str
    width: float = 1.0  # 默认宽度 1um
    length: float = 0.1  # 默认长度 100nm
    model: str = ""


@dataclass
class Circuit:
    """
    电路数据结构
    
    属性:
        name: 电路名称
        devices: MOS器件字典 {器件名: MOSDevice}
        inputs: 输入端口列表
        outputs: 输出端口列表
        nodes: 所有节点集合
        vdd_node: 电源节点名
        gnd_node: 地节点名
        node_connections: 节点连接关系 {节点名: [(器件名, 端口类型)]}
    """
    name: str = ""
    devices: Dict[str, MOSDevice] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    nodes: Set[str] = field(default_factory=set)
    vdd_node: str = "VDD"
    gnd_node: str = "GND"
    node_connections: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)


def parse_spice_netlist(netlist_content: str) -> Circuit:
    """
    解析SPICE网表内容，提取电路信息
    
    参数:
        netlist_content: SPICE网表字符串
        
    返回:
        Circuit: 解析后的电路对象
    """
    circuit = Circuit()
    lines = netlist_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 跳过注释和空行
        if not line or line.startswith('*') or line.startswith('//'):
            continue
            
        # 解析子电路定义 .subckt
        if line.lower().startswith('.subckt'):
            parts = line.split()
            circuit.name = parts[1]
            # 端口定义在子电路名称之后
            ports = parts[2:]
            # 通常第一个是输出，后面是输入，最后两个是VDD和GND
            # 但这可能因电路而异，我们先全部记录
            for port in ports:
                circuit.nodes.add(port)
            continue
            
        # 解析输入/输出端口声明
        if line.lower().startswith('.inputs'):
            parts = line.split()[1:]
            circuit.inputs.extend(parts)
            continue
            
        if line.lower().startswith('.outputs'):
            parts = line.split()[1:]
            circuit.outputs.extend(parts)
            continue
            
        # 解析MOS器件 (M开头)
        # 格式: Mname drain gate source bulk model_name W=xxx L=xxx
        if line.upper().startswith('M'):
            device = parse_mos_line(line)
            if device:
                circuit.devices[device.name] = device
                # 记录节点
                for node in [device.drain, device.gate, device.source, device.bulk]:
                    circuit.nodes.add(node)
                    if node not in circuit.node_connections:
                        circuit.node_connections[node] = []
                circuit.node_connections[device.drain].append((device.name, 'D'))
                circuit.node_connections[device.gate].append((device.name, 'G'))
                circuit.node_connections[device.source].append((device.name, 'S'))
                circuit.node_connections[device.bulk].append((device.name, 'B'))
            continue
    
    # 自动识别VDD和GND节点
    for node in circuit.nodes:
        node_upper = node.upper()
        if node_upper in ['VDD', 'VDDD', 'VCC', 'SUPPLY']:
            circuit.vdd_node = node
        elif node_upper in ['GND', 'VSS', 'GROUND', '0']:
            circuit.gnd_node = node
    
    return circuit


def parse_mos_line(line: str) -> Optional[MOSDevice]:
    """
    解析单行MOS器件定义
    
    参数:
        line: MOS器件定义行
        
    返回:
        MOSDevice: 解析后的MOS器件对象，解析失败返回None
    
    示例输入格式:
        MN1 out in gnd gnd nmos W=1u L=0.1u
        MP1 out in vdd vdd pmos W=2u L=0.1u
    """
    parts = line.split()
    if len(parts) < 6:
        return None
    
    name = parts[0]
    drain = parts[1]
    gate = parts[2]
    source = parts[3]
    bulk = parts[4]
    model = parts[5] if len(parts) > 5 else ""
    
    # 判断MOS类型
    model_lower = model.lower()
    if 'nmos' in model_lower or name.upper().startswith('MN'):
        mos_type = MOSType.NMOS
    elif 'pmos' in model_lower or name.upper().startswith('MP'):
        mos_type = MOSType.PMOS
    else:
        # 根据名称推断
        if 'N' in name.upper():
            mos_type = MOSType.NMOS
        else:
            mos_type = MOSType.PMOS
    
    # 解析宽度和长度
    width = 1.0
    length = 0.1
    for part in parts[6:]:
        if part.upper().startswith('W='):
            width = parse_unit_value(part[2:])
        elif part.upper().startswith('L='):
            length = parse_unit_value(part[2:])
    
    return MOSDevice(
        name=name,
        mos_type=mos_type,
        drain=drain,
        gate=gate,
        source=source,
        bulk=bulk,
        width=width,
        length=length,
        model=model
    )


def parse_unit_value(value_str: str) -> float:
    """
    解析带单位的数值
    
    参数:
        value_str: 带单位的数值字符串 (如 "1u", "100n", "0.1")
        
    返回:
        float: 转换后的浮点数值 (单位统一为um)
    """
    value_str = value_str.strip().lower()
    
    # 单位转换表 (转换为um)
    unit_multipliers = {
        'u': 1.0,          # 微米
        'um': 1.0,
        'n': 0.001,        # 纳米 -> 微米
        'nm': 0.001,
        'p': 0.000001,     # 皮米 -> 微米
        'm': 1000.0,       # 毫米 -> 微米
        'mm': 1000.0,
    }
    
    # 尝试提取数值和单位
    match = re.match(r'([0-9.e+-]+)\s*([a-z]*)', value_str)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        multiplier = unit_multipliers.get(unit, 1.0)
        return num * multiplier
    
    return float(value_str)


def get_connected_devices(circuit: Circuit, node: str) -> List[Tuple[str, str]]:
    """
    获取连接到指定节点的所有器件
    
    参数:
        circuit: 电路对象
        node: 节点名称
        
    返回:
        List[Tuple[str, str]]: 器件列表 [(器件名, 连接端口)]
    """
    return circuit.node_connections.get(node, [])


def find_path_devices(circuit: Circuit, start_node: str, end_node: str) -> List[str]:
    """
    找到从start_node到end_node路径上的所有MOS器件
    使用BFS搜索
    
    参数:
        circuit: 电路对象
        start_node: 起始节点
        end_node: 终止节点
        
    返回:
        List[str]: 路径上的器件名称列表
    """
    from collections import deque
    
    visited = set()
    queue = deque([(start_node, [])])
    all_path_devices = set()
    
    while queue:
        current_node, path_devices = queue.popleft()
        
        if current_node == end_node:
            all_path_devices.update(path_devices)
            continue
            
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # 遍历连接到当前节点的所有器件
        for device_name, terminal in circuit.node_connections.get(current_node, []):
            device = circuit.devices[device_name]
            
            # 通过D-S通道找到下一个节点
            if terminal == 'D':
                next_node = device.source
            elif terminal == 'S':
                next_node = device.drain
            else:
                continue
            
            new_path = path_devices + [device_name]
            queue.append((next_node, new_path))
    
    return list(all_path_devices)


def print_circuit_info(circuit: Circuit):
    """打印电路信息摘要"""
    print(f"\n{'='*50}")
    print(f"电路名称: {circuit.name}")
    print(f"{'='*50}")
    print(f"输入端口: {circuit.inputs}")
    print(f"输出端口: {circuit.outputs}")
    print(f"VDD节点: {circuit.vdd_node}")
    print(f"GND节点: {circuit.gnd_node}")
    print(f"\nMOS器件数量: {len(circuit.devices)}")
    print("-"*50)
    
    for name, device in circuit.devices.items():
        print(f"  {name}: {device.mos_type.value.upper()}")
        print(f"    D={device.drain}, G={device.gate}, S={device.source}")
        print(f"    W={device.width}um, L={device.length}um")
    
    print(f"\n节点数量: {len(circuit.nodes)}")
    print(f"节点列表: {circuit.nodes}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 测试用例：两级反相器
    test_netlist = """
    * Two-stage Inverter
    .subckt INV2 OUT IN VDD GND
    .inputs IN
    .outputs OUT
    
    * 第一级反相器
    MN1 MID IN GND GND nmos W=1u L=0.1u
    MP1 MID IN VDD VDD pmos W=2u L=0.1u
    
    * 第二级反相器  
    MN2 OUT MID GND GND nmos W=1u L=0.1u
    MP2 OUT MID VDD VDD pmos W=2u L=0.1u
    
    .ends INV2
    """
    
    circuit = parse_spice_netlist(test_netlist)
    print_circuit_info(circuit)
