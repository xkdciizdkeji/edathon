"""
Dominant Device 识别算法 (Dominant Device Finder)

该模块实现了识别电路中对延时影响最大的MOS器件(dominant devices)的算法。

核心思想:
1. 解析电路拓扑结构
2. 根据输入信号的跳变方向，分析每个节点的电平变化
3. 识别在当前跳变条件下"活跃"(conducting)的MOS管
4. 通过电流路径分析和敏感性评估，计算每个MOS管对延时的贡献
5. 按贡献度排序，筛选出dominant devices

算法原理:
-----------
1. 对于NMOS: 当gate为高电平时导通
   - 导通时: 漏极电流 Id ∝ (Vgs - Vth)² (饱和区)
   - 影响放电路径(pull-down)的延时
   
2. 对于PMOS: 当gate为低电平时导通
   - 导通时: 漏极电流 Id ∝ (Vsg - |Vth|)² (饱和区)
   - 影响充电路径(pull-up)的延时

3. dominant devices是指在当前输入跳变下，处于导通状态且在信号传播路径上的MOS管

"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from circuit_parser import Circuit, MOSDevice, MOSType, parse_spice_netlist


class NodeState(Enum):
    """节点电平状态"""
    LOW = 0      # 低电平 (接近GND)
    HIGH = 1     # 高电平 (接近VDD)
    UNKNOWN = -1 # 未知状态
    RISING = 2   # 上升沿 (0→1)
    FALLING = 3  # 下降沿 (1→0)


@dataclass
class DeviceAnalysis:
    """
    MOS器件分析结果
    
    属性:
        name: 器件名称
        mos_type: MOS类型
        is_conducting: 是否处于导通状态
        gate_state: 栅极电平状态
        drain_transition: 漏极电平变化
        on_critical_path: 是否在关键路径上
        sensitivity_score: 敏感性得分 (对延时的影响程度)
        contribution_reason: 贡献原因说明
    """
    name: str
    mos_type: MOSType
    is_conducting: bool = False
    gate_state: NodeState = NodeState.UNKNOWN
    drain_transition: NodeState = NodeState.UNKNOWN
    on_critical_path: bool = False
    sensitivity_score: float = 0.0
    contribution_reason: str = ""


class DominantDeviceFinder:
    """
    Dominant Device 识别器
    
    该类实现了识别电路中dominant devices的完整算法流程。
    """
    
    def __init__(self, circuit: Circuit):
        """
        初始化识别器
        
        参数:
            circuit: 解析后的电路对象
        """
        self.circuit = circuit
        self.node_states: Dict[str, NodeState] = {}
        self.device_analysis: Dict[str, DeviceAnalysis] = {}
        
        # 初始化电源和地的状态
        self.node_states[circuit.vdd_node] = NodeState.HIGH
        self.node_states[circuit.gnd_node] = NodeState.LOW
    
    def analyze(
        self, 
        input_transitions: Dict[str, str],
        initial_states: Dict[str, str] = None
    ) -> List[Tuple[str, float]]:
        """
        分析电路，识别dominant devices
        
        参数:
            input_transitions: 输入跳变信息 {输入名: "rise"/"fall"}
            initial_states: 初始节点状态 {节点名: "high"/"low"}, 可选
            
        返回:
            List[Tuple[str, float]]: 排序后的dominant devices列表
                                    [(器件名, 敏感性得分)]
        """
        # Step 1: 初始化节点状态
        self._initialize_node_states(input_transitions, initial_states)
        
        # Step 2: 传播信号，确定所有节点的状态变化
        self._propagate_signals()
        
        # Step 3: 分析每个MOS器件的状态
        self._analyze_devices()
        
        # Step 4: 识别关键路径上的器件
        self._identify_critical_path()
        
        # Step 5: 计算每个器件的敏感性得分
        self._calculate_sensitivity_scores()
        
        # Step 6: 返回排序后的dominant devices
        return self._get_dominant_devices()
    
    def _initialize_node_states(
        self, 
        input_transitions: Dict[str, str],
        initial_states: Dict[str, str] = None
    ):
        """
        初始化节点状态
        
        根据输入跳变信息设置输入节点的状态变化。
        
        参数:
            input_transitions: 输入跳变信息
            initial_states: 初始状态(可选)
        """
        # 设置电源和地
        self.node_states[self.circuit.vdd_node] = NodeState.HIGH
        self.node_states[self.circuit.gnd_node] = NodeState.LOW
        
        # 设置初始状态
        if initial_states:
            for node, state in initial_states.items():
                if state.lower() == "high":
                    self.node_states[node] = NodeState.HIGH
                elif state.lower() == "low":
                    self.node_states[node] = NodeState.LOW
        
        # 设置输入跳变
        for input_name, transition in input_transitions.items():
            if transition.lower() == "rise":
                self.node_states[input_name] = NodeState.RISING
            elif transition.lower() == "fall":
                self.node_states[input_name] = NodeState.FALLING
    
    def _propagate_signals(self):
        """
        信号传播分析
        
        使用迭代方法传播信号，确定每个内部节点的状态变化。
        这是一个简化的逻辑仿真，假设稳态下的行为。
        
        算法说明:
        ---------
        1. 从已知状态的节点开始
        2. 对于每个MOS管，根据其gate状态判断是否导通
        3. 如果导通，则source和drain之间有电流路径
        4. 根据pull-up和pull-down网络的导通情况，确定输出节点的状态
        5. 迭代直到所有节点状态稳定
        """
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            changed = False
            iteration += 1
            
            # 遍历所有内部节点
            for node in self.circuit.nodes:
                if node in [self.circuit.vdd_node, self.circuit.gnd_node]:
                    continue
                if node in self.circuit.inputs:
                    continue
                if node in self.node_states and self.node_states[node] != NodeState.UNKNOWN:
                    continue
                
                # 分析连接到该节点的MOS管
                new_state = self._analyze_node_state(node)
                if new_state != NodeState.UNKNOWN:
                    if node not in self.node_states or self.node_states[node] != new_state:
                        self.node_states[node] = new_state
                        changed = True
            
            if not changed:
                break
    
    def _analyze_node_state(self, node: str) -> NodeState:
        """
        分析单个节点的状态
        
        参数:
            node: 节点名称
            
        返回:
            NodeState: 节点状态
        """
        connected_devices = self.circuit.node_connections.get(node, [])
        
        has_pullup = False    # 是否有导通的上拉路径
        has_pulldown = False  # 是否有导通的下拉路径
        
        for device_name, terminal in connected_devices:
            if terminal not in ['D', 'S']:
                continue
            
            device = self.circuit.devices[device_name]
            gate_state = self.node_states.get(device.gate, NodeState.UNKNOWN)
            
            # 判断MOS是否导通
            is_conducting = self._is_device_conducting(device, gate_state)
            
            if not is_conducting:
                continue
            
            # 获取另一端连接的节点
            other_terminal = device.source if terminal == 'D' else device.drain
            other_state = self.node_states.get(other_terminal, NodeState.UNKNOWN)
            
            # 判断是上拉还是下拉
            if other_state == NodeState.HIGH or other_terminal == self.circuit.vdd_node:
                has_pullup = True
            elif other_state == NodeState.LOW or other_terminal == self.circuit.gnd_node:
                has_pulldown = True
            elif other_state == NodeState.RISING:
                has_pullup = True
            elif other_state == NodeState.FALLING:
                has_pulldown = True
        
        # 根据上拉/下拉情况确定节点状态
        if has_pullup and not has_pulldown:
            return NodeState.RISING
        elif has_pulldown and not has_pullup:
            return NodeState.FALLING
        elif has_pullup and has_pulldown:
            # 竞争状态，需要更复杂的分析
            return NodeState.UNKNOWN
        
        return NodeState.UNKNOWN
    
    def _is_device_conducting(self, device: MOSDevice, gate_state: NodeState) -> bool:
        """
        判断MOS器件是否处于导通状态
        
        规则:
        - NMOS: gate为高电平(HIGH)或上升沿(RISING)时导通
        - PMOS: gate为低电平(LOW)或下降沿(FALLING)时导通
        
        参数:
            device: MOS器件对象
            gate_state: 栅极电平状态
            
        返回:
            bool: 是否导通
        """
        if device.mos_type == MOSType.NMOS:
            # NMOS: gate高时导通
            return gate_state in [NodeState.HIGH, NodeState.RISING]
        else:
            # PMOS: gate低时导通
            return gate_state in [NodeState.LOW, NodeState.FALLING]
    
    def _analyze_devices(self):
        """
        分析所有MOS器件的状态
        
        对每个MOS管进行详细分析，包括:
        - 导通状态
        - 栅极状态
        - 漏极变化
        """
        for name, device in self.circuit.devices.items():
            gate_state = self.node_states.get(device.gate, NodeState.UNKNOWN)
            drain_state = self.node_states.get(device.drain, NodeState.UNKNOWN)
            source_state = self.node_states.get(device.source, NodeState.UNKNOWN)
            
            is_conducting = self._is_device_conducting(device, gate_state)
            
            # 确定漏极的变化方向
            drain_transition = drain_state
            
            # 生成分析结果
            analysis = DeviceAnalysis(
                name=name,
                mos_type=device.mos_type,
                is_conducting=is_conducting,
                gate_state=gate_state,
                drain_transition=drain_transition,
                on_critical_path=False,  # 后续分析
                sensitivity_score=0.0,    # 后续计算
                contribution_reason=""
            )
            
            self.device_analysis[name] = analysis
    
    def _identify_critical_path(self):
        """
        识别关键路径上的器件
        
        关键路径是从输入到输出的信号传播路径。
        在这条路径上的导通MOS管对延时有直接影响。
        
        算法:
        -----
        1. 从输出节点开始反向追踪
        2. 找到所有连接到输出且导通的MOS管
        3. 继续追踪这些MOS管的输入端
        4. 直到到达输入节点或电源/地
        """
        visited = set()
        
        # 从每个输出节点开始追踪
        for output in self.circuit.outputs:
            self._trace_critical_path_recursive(output, visited)
    
    def _trace_critical_path_recursive(self, node: str, visited: Set[str]):
        """
        递归追踪关键路径
        
        参数:
            node: 当前节点
            visited: 已访问节点集合
        """
        if node in visited:
            return
        visited.add(node)
        
        # 电源和地是终点
        if node in [self.circuit.vdd_node, self.circuit.gnd_node]:
            return
        
        # 获取连接到该节点的器件
        connected = self.circuit.node_connections.get(node, [])
        
        for device_name, terminal in connected:
            # 只考虑通过D/S连接的器件
            if terminal not in ['D', 'S']:
                continue
            
            analysis = self.device_analysis.get(device_name)
            if not analysis:
                continue
            
            # 只有导通的器件才在关键路径上
            if analysis.is_conducting:
                analysis.on_critical_path = True
                
                # 继续追踪另一端
                device = self.circuit.devices[device_name]
                other_node = device.source if terminal == 'D' else device.drain
                self._trace_critical_path_recursive(other_node, visited)
            
            # 也追踪栅极连接(如果栅极正在变化)
            device = self.circuit.devices[device_name]
            if device.gate not in [self.circuit.vdd_node, self.circuit.gnd_node]:
                self._trace_critical_path_recursive(device.gate, visited)
    
    def _calculate_sensitivity_scores(self):
        """
        计算每个器件的敏感性得分
        
        敏感性得分反映了该器件对总体延时的影响程度。
        
        评分标准:
        ---------
        1. 基础分: 导通的器件获得基础分
        2. 路径分: 在关键路径上的器件获得额外分数
        3. 位置分: 
           - 直接驱动输出的器件得分更高
           - 第一级(靠近输入)的器件得分也较高
        4. 尺寸分: W/L比值影响电流驱动能力
        5. 类型匹配分:
           - 如果输出下降，pull-down网络中的NMOS更重要
           - 如果输出上升，pull-up网络中的PMOS更重要
        """
        for name, analysis in self.device_analysis.items():
            score = 0.0
            reasons = []
            
            device = self.circuit.devices[name]
            
            # 1. 导通状态基础分 (最重要)
            if analysis.is_conducting:
                score += 50.0
                reasons.append("导通状态")
            else:
                # 非导通器件得分很低
                analysis.sensitivity_score = 0.0
                analysis.contribution_reason = "非导通状态"
                continue
            
            # 2. 关键路径分
            if analysis.on_critical_path:
                score += 30.0
                reasons.append("关键路径")
            
            # 3. 位置分析
            # 检查是否直接连接到输出
            for output in self.circuit.outputs:
                if device.drain == output or device.source == output:
                    score += 15.0
                    reasons.append("直接驱动输出")
                    break
            
            # 检查是否直接连接到输入
            for input_node in self.circuit.inputs:
                if device.gate == input_node:
                    score += 10.0
                    reasons.append("第一级器件")
                    break
            
            # 4. 类型匹配分
            # 获取输出节点的变化方向
            for output in self.circuit.outputs:
                output_state = self.node_states.get(output, NodeState.UNKNOWN)
                
                # 输出下降时，NMOS(放电)更重要
                if output_state == NodeState.FALLING:
                    if device.mos_type == MOSType.NMOS:
                        score += 20.0
                        reasons.append("放电路径(NMOS)")
                # 输出上升时，PMOS(充电)更重要
                elif output_state == NodeState.RISING:
                    if device.mos_type == MOSType.PMOS:
                        score += 20.0
                        reasons.append("充电路径(PMOS)")
            
            # 5. 尺寸因素 (W/L比值影响驱动能力)
            wl_ratio = device.width / device.length if device.length > 0 else 10.0
            size_factor = min(wl_ratio / 10.0, 2.0)  # 归一化
            score += size_factor * 5.0
            
            analysis.sensitivity_score = score
            analysis.contribution_reason = ", ".join(reasons)
    
    def _get_dominant_devices(self, threshold: float = 50.0) -> List[Tuple[str, float]]:
        """
        获取dominant devices列表
        
        参数:
            threshold: 敏感性得分阈值，只返回超过阈值的器件
            
        返回:
            List[Tuple[str, float]]: 排序后的器件列表 [(器件名, 得分)]
        """
        # 按敏感性得分排序
        ranked = [
            (name, analysis.sensitivity_score)
            for name, analysis in self.device_analysis.items()
            if analysis.sensitivity_score >= threshold
        ]
        
        # 降序排列
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def get_analysis_report(self) -> str:
        """
        生成详细的分析报告
        
        返回:
            str: 格式化的分析报告
        """
        report = []
        report.append("=" * 60)
        report.append("Dominant Device 分析报告")
        report.append("=" * 60)
        
        # 节点状态
        report.append("\n[节点状态]")
        for node, state in sorted(self.node_states.items()):
            report.append(f"  {node}: {state.name}")
        
        # 器件分析
        report.append("\n[器件分析]")
        
        # 按得分排序
        sorted_devices = sorted(
            self.device_analysis.items(),
            key=lambda x: x[1].sensitivity_score,
            reverse=True
        )
        
        for name, analysis in sorted_devices:
            device = self.circuit.devices[name]
            report.append(f"\n  {name} ({device.mos_type.value.upper()}):")
            report.append(f"    导通状态: {'是' if analysis.is_conducting else '否'}")
            report.append(f"    关键路径: {'是' if analysis.on_critical_path else '否'}")
            report.append(f"    栅极状态: {analysis.gate_state.name}")
            report.append(f"    敏感性得分: {analysis.sensitivity_score:.1f}")
            if analysis.contribution_reason:
                report.append(f"    贡献原因: {analysis.contribution_reason}")
        
        # Dominant devices
        dominant = self._get_dominant_devices()
        report.append("\n" + "=" * 60)
        report.append("[识别的 Dominant Devices]")
        report.append("=" * 60)
        for rank, (name, score) in enumerate(dominant, 1):
            report.append(f"  {rank}. {name} (得分: {score:.1f})")
        
        return "\n".join(report)


def find_dominant_devices(
    netlist: str,
    input_transitions: Dict[str, str],
    initial_states: Dict[str, str] = None,
    verbose: bool = True
) -> List[str]:
    """
    便捷函数：识别电路中的dominant devices
    
    参数:
        netlist: SPICE网表字符串
        input_transitions: 输入跳变信息 {输入名: "rise"/"fall"}
        initial_states: 初始节点状态(可选)
        verbose: 是否打印详细报告
        
    返回:
        List[str]: dominant devices名称列表
    """
    # 解析电路
    circuit = parse_spice_netlist(netlist)
    
    # 创建分析器
    finder = DominantDeviceFinder(circuit)
    
    # 执行分析
    ranked_devices = finder.analyze(input_transitions, initial_states)
    
    # 打印报告
    if verbose:
        print(finder.get_analysis_report())
    
    # 返回器件名称列表
    return [name for name, score in ranked_devices]


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 测试: 两级反相器
    test_netlist = """
    * Two-stage Inverter
    .subckt INV2 OUT IN VDD GND
    .inputs IN
    .outputs OUT
    
    * Stage 1
    MN1 MID IN GND GND nmos W=1u L=0.1u
    MP1 MID IN VDD VDD pmos W=2u L=0.1u
    
    * Stage 2
    MN2 OUT MID GND GND nmos W=1u L=0.1u
    MP2 OUT MID VDD VDD pmos W=2u L=0.1u
    
    .ends INV2
    """
    
    print("测试: 两级反相器，输入 0→1")
    print("预期结果: MN1和MP2为dominant devices")
    print()
    
    dominant = find_dominant_devices(
        test_netlist,
        input_transitions={"IN": "rise"},
        verbose=True
    )
    
    print(f"\n最终识别的dominant devices: {dominant}")
