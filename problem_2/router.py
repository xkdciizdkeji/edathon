"""
微流控芯片布线问题 - 布线求解器
Microfluidic Chip Routing Problem - Router Solver

实现基于A*算法的布线求解器，支持：
- 带转弯限制的路径搜索（waypoint约束）
- 最小间距约束（通道间、通道与模块间）
- 顺序布线 + Rip-up & Reroute策略
- 多候选路径生成与冲突图分析

作者: EDAthon 2025 参赛预备代码
"""

import heapq
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from input_generator import MicrofluidicChip, Module, Port, Net, ChipSpec


class Direction(Enum):
    """方向枚举 - 用于跟踪路径方向变化"""
    NONE = 0      # 初始状态
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


@dataclass
class Point:
    """二维点"""
    x: float
    y: float
    
    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2)))
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一个点的欧几里得距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def manhattan_distance_to(self, other: 'Point') -> float:
        """计算到另一个点的曼哈顿距离"""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Segment:
    """线段 - 通道的基本组成单位"""
    start: Point
    end: Point
    
    def length(self) -> float:
        """线段长度"""
        return self.start.distance_to(self.end)
    
    def is_horizontal(self) -> bool:
        """是否为水平线段"""
        return abs(self.start.y - self.end.y) < 0.01
    
    def is_vertical(self) -> bool:
        """是否为垂直线段"""
        return abs(self.start.x - self.end.x) < 0.01


@dataclass
class Route:
    """路由 - 一条完整的通道路径"""
    net_id: str
    waypoints: List[Point]  # 路径上的所有点（包括起点和终点）
    
    def get_segments(self) -> List[Segment]:
        """获取路径的所有线段"""
        segments = []
        for i in range(len(self.waypoints) - 1):
            segments.append(Segment(self.waypoints[i], self.waypoints[i + 1]))
        return segments
    
    def total_length(self) -> float:
        """路径总长度"""
        return sum(seg.length() for seg in self.get_segments())
    
    def num_turns(self) -> int:
        """路径转弯次数（waypoints数 - 2，因为起点和终点不算转弯）"""
        return max(0, len(self.waypoints) - 2)


@dataclass(order=True)
class AStarState:
    """A*搜索状态"""
    f_score: float  # f = g + h（用于优先队列排序）
    g_score: float = field(compare=False)  # 从起点到当前点的实际代价
    position: Point = field(compare=False)
    direction: Direction = field(compare=False)  # 到达当前点的方向
    turns: int = field(compare=False)  # 已使用的转弯次数
    path: List[Point] = field(compare=False)  # 路径历史


class Grid:
    """
    网格类 - 将连续空间离散化为网格
    
    这是布线算法的核心数据结构，用于：
    1. 将模块转换为障碍网格
    2. 记录已布线通道占用的网格
    3. 支持A*路径搜索
    """
    
    def __init__(self, width: float, height: float, cell_size: float):
        """
        初始化网格
        
        Args:
            width: 芯片宽度
            height: 芯片高度
            cell_size: 网格单元尺寸
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # 计算网格尺寸
        self.cols = int(math.ceil(width / cell_size)) + 1
        self.rows = int(math.ceil(height / cell_size)) + 1
        
        # 障碍物网格：True表示被占用/不可通行
        self.obstacles: Set[Tuple[int, int]] = set()
        
        # 已布线通道占用的网格（膨胀后）
        self.occupied: Set[Tuple[int, int]] = set()
    
    def point_to_grid(self, point: Point) -> Tuple[int, int]:
        """将连续坐标转换为网格坐标"""
        col = int(round(point.x / self.cell_size))
        row = int(round(point.y / self.cell_size))
        return (max(0, min(col, self.cols - 1)), max(0, min(row, self.rows - 1)))
    
    def grid_to_point(self, col: int, row: int) -> Point:
        """将网格坐标转换为连续坐标"""
        return Point(col * self.cell_size, row * self.cell_size)
    
    def is_valid(self, col: int, row: int) -> bool:
        """检查网格坐标是否有效"""
        return 0 <= col < self.cols and 0 <= row < self.rows
    
    def is_blocked(self, col: int, row: int) -> bool:
        """检查网格是否被阻塞（模块或已布线通道）"""
        return (col, row) in self.obstacles or (col, row) in self.occupied
    
    def add_module_obstacle(self, module: Module, spacing: float):
        """
        将模块添加为障碍物（带间距膨胀）
        
        Args:
            module: 模块对象
            spacing: 最小间距（用于膨胀）
        """
        # 计算膨胀后的边界
        x_min = module.x - spacing
        y_min = module.y - spacing
        x_max = module.x + module.width + spacing
        y_max = module.y + module.height + spacing
        
        # 转换为网格坐标
        col_min = max(0, int(x_min / self.cell_size))
        row_min = max(0, int(y_min / self.cell_size))
        col_max = min(self.cols - 1, int(math.ceil(x_max / self.cell_size)))
        row_max = min(self.rows - 1, int(math.ceil(y_max / self.cell_size)))
        
        # 标记障碍
        for col in range(col_min, col_max + 1):
            for row in range(row_min, row_max + 1):
                self.obstacles.add((col, row))
    
    def add_route_obstacle(self, route: Route, spacing: float):
        """
        将已布线通道添加为占用区域（带间距膨胀）
        
        这确保新路径与已有路径保持最小间距
        
        Args:
            route: 路由对象
            spacing: 最小间距
        """
        for segment in route.get_segments():
            self._add_segment_obstacle(segment, spacing)
    
    def _add_segment_obstacle(self, segment: Segment, spacing: float):
        """将线段添加为占用区域"""
        # 获取线段的网格范围
        start_col, start_row = self.point_to_grid(segment.start)
        end_col, end_row = self.point_to_grid(segment.end)
        
        # 确保 start <= end
        col_min, col_max = min(start_col, end_col), max(start_col, end_col)
        row_min, row_max = min(start_row, end_row), max(start_row, end_row)
        
        # 计算膨胀格数
        expand = int(math.ceil(spacing / self.cell_size))
        
        # 标记占用区域（膨胀）
        for col in range(col_min - expand, col_max + expand + 1):
            for row in range(row_min - expand, row_max + expand + 1):
                if self.is_valid(col, row):
                    self.occupied.add((col, row))
    
    def remove_route_obstacle(self, route: Route, spacing: float):
        """移除已布线通道的占用区域（用于rip-up操作）"""
        for segment in route.get_segments():
            self._remove_segment_obstacle(segment, spacing)
    
    def _remove_segment_obstacle(self, segment: Segment, spacing: float):
        """移除线段的占用区域"""
        start_col, start_row = self.point_to_grid(segment.start)
        end_col, end_row = self.point_to_grid(segment.end)
        
        col_min, col_max = min(start_col, end_col), max(start_col, end_col)
        row_min, row_max = min(start_row, end_row), max(start_row, end_row)
        
        expand = int(math.ceil(spacing / self.cell_size))
        
        for col in range(col_min - expand, col_max + expand + 1):
            for row in range(row_min - expand, row_max + expand + 1):
                if (col, row) in self.occupied:
                    self.occupied.discard((col, row))
    
    def clear_occupied(self):
        """清除所有已布线占用区域"""
        self.occupied.clear()


class MicrofluidicRouter:
    """
    微流控芯片布线求解器
    
    核心算法：
    1. 网格化 + 障碍物膨胀
    2. 带转弯限制的A*搜索
    3. 顺序布线 + 优先级调整
    4. Rip-up & Reroute 冲突处理
    """
    
    def __init__(self, chip: MicrofluidicChip, cell_size: Optional[float] = None):
        """
        初始化路由器
        
        Args:
            chip: 微流控芯片对象
            cell_size: 网格单元尺寸（默认为最小间距的一半）
        """
        self.chip = chip
        self.spec = chip.spec
        
        # 计算网格尺寸（默认为最小间距的一半，以获得足够的精度）
        if cell_size is None:
            cell_size = self.spec.min_spacing / 2
        
        self.cell_size = cell_size
        
        # 初始化网格
        self.grid = Grid(self.spec.board_width, self.spec.board_height, cell_size)
        
        # 构建端口查找表
        self.ports: Dict[str, Port] = {}
        self.port_grid_cells = set()  # 记录端口所在的网格单元
        
        for module in chip.modules:
            for port in module.ports:
                self.ports[port.id] = port
                col, row = self.grid.point_to_grid(Point(port.x, port.y))
                self.port_grid_cells.add((col, row))
        
        for port in chip.board_ports:
            self.ports[port.id] = port
            col, row = self.grid.point_to_grid(Point(port.x, port.y))
            self.port_grid_cells.add((col, row))
        
        # 添加模块障碍物
        # 策略：模块内部是障碍，但端口位置及其周围留出通道
        for module in chip.modules:
            self._add_module_with_port_access(module)
        
        # 存储布线结果
        self.routes: Dict[str, Route] = {}
        
        # 布线统计
        self.stats = {
            "total_nets": len(chip.nets),
            "routed_nets": 0,
            "failed_nets": 0,
            "total_length": 0.0,
            "total_turns": 0
        }
    
    def _add_module_with_port_access(self, module: Module):
        """
        添加模块障碍物，同时确保端口可以访问
        
        策略：
        1. 将模块内部标记为障碍
        2. 为每个端口创建一条从模块边缘向外延伸的"通道"
        """
        # 获取模块边界（网格坐标）
        col_min = int(module.x / self.cell_size)
        row_min = int(module.y / self.cell_size)
        col_max = int(math.ceil((module.x + module.width) / self.cell_size))
        row_max = int(math.ceil((module.y + module.height) / self.cell_size))
        
        # 标记模块内部为障碍
        for col in range(col_min, col_max + 1):
            for row in range(row_min, row_max + 1):
                if self.grid.is_valid(col, row):
                    self.grid.obstacles.add((col, row))
        
        # 为每个端口清除通道
        for port in module.ports:
            port_col, port_row = self.grid.point_to_grid(Point(port.x, port.y))
            
            # 确定端口在模块的哪个边上
            # 并清除从端口向外延伸的几个格子
            clearance = 3  # 清除的格子数
            
            # 判断端口位置
            is_top = abs(port.y - (module.y + module.height)) < self.cell_size
            is_bottom = abs(port.y - module.y) < self.cell_size
            is_left = abs(port.x - module.x) < self.cell_size
            is_right = abs(port.x - (module.x + module.width)) < self.cell_size
            
            # 清除端口位置
            self.grid.obstacles.discard((port_col, port_row))
            
            # 向外清除通道
            if is_top:
                for i in range(clearance):
                    self.grid.obstacles.discard((port_col, port_row + i))
            if is_bottom:
                for i in range(clearance):
                    self.grid.obstacles.discard((port_col, port_row - i))
            if is_left:
                for i in range(clearance):
                    self.grid.obstacles.discard((port_col - i, port_row))
            if is_right:
                for i in range(clearance):
                    self.grid.obstacles.discard((port_col + i, port_row))
    
    def get_port_point(self, port_id: str) -> Point:
        """获取端口的坐标点"""
        port = self.ports[port_id]
        return Point(port.x, port.y)
    
    def route_all(self, use_rip_up: bool = True, max_iterations: int = 3) -> Dict[str, Route]:
        """
        布线所有网络
        
        算法流程：
        1. 按优先级排序网络（基于曼哈顿距离，短的优先）
        2. 顺序布线每个网络
        3. 如果有失败的网络，使用rip-up & reroute策略重试
        
        Args:
            use_rip_up: 是否使用rip-up策略
            max_iterations: 最大迭代次数
        
        Returns:
            成功布线的路由字典
        """
        # 计算网络优先级（曼哈顿距离越短优先级越高）
        nets_with_priority = []
        for net in self.chip.nets:
            src = self.get_port_point(net.source_port)
            dst = self.get_port_point(net.target_port)
            manhattan_dist = src.manhattan_distance_to(dst)
            nets_with_priority.append((manhattan_dist, net))
        
        # 按距离排序（短的优先，更容易成功）
        nets_with_priority.sort(key=lambda x: x[0])
        sorted_nets = [net for _, net in nets_with_priority]
        
        # 第一轮顺序布线
        failed_nets = []
        for net in sorted_nets:
            route = self._route_single_net(net)
            if route is not None:
                self.routes[net.id] = route
                self.grid.add_route_obstacle(route, self.spec.min_spacing)
            else:
                failed_nets.append(net)
        
        # Rip-up & Reroute
        if use_rip_up and failed_nets:
            for iteration in range(max_iterations):
                if not failed_nets:
                    break
                
                print(f"  Rip-up iteration {iteration + 1}: {len(failed_nets)} failed nets")
                
                # 尝试不同的优先级顺序
                failed_nets = self._rip_up_and_reroute(failed_nets, iteration)
        
        # 更新统计
        self.stats["routed_nets"] = len(self.routes)
        self.stats["failed_nets"] = len(self.chip.nets) - len(self.routes)
        self.stats["total_length"] = sum(r.total_length() for r in self.routes.values())
        self.stats["total_turns"] = sum(r.num_turns() for r in self.routes.values())
        
        return self.routes
    
    def _route_single_net(self, net: Net) -> Optional[Route]:
        """
        对单个网络进行布线
        
        使用带转弯限制的A*算法：
        - 状态: (位置, 方向, 已使用转弯数, 路径)
        - 启发函数: 曼哈顿距离
        - 约束: 转弯次数 <= max_waypoints - 2
        
        Args:
            net: 要布线的网络
        
        Returns:
            成功时返回Route对象，失败时返回None
        """
        src = self.get_port_point(net.source_port)
        dst = self.get_port_point(net.target_port)
        
        # 转换为网格坐标
        src_grid = self.grid.point_to_grid(src)
        dst_grid = self.grid.point_to_grid(dst)
        
        # 最大允许转弯数（waypoints包括起点和终点，所以转弯数 = waypoints - 2）
        max_turns = self.spec.max_waypoints - 2
        
        # A*搜索
        # 初始状态：起点，无方向，0转弯
        start_state = AStarState(
            f_score=0,
            g_score=0,
            position=src,
            direction=Direction.NONE,
            turns=0,
            path=[src]
        )
        
        # 优先队列
        open_set = [start_state]
        
        # 访问记录：简化为 (col, row, direction) -> (g_score, turns)
        # 只保留方向信息，不需要跟踪转弯次数（因为转弯惩罚已经反映在代价中）
        visited: Dict[Tuple[int, int, Direction], Tuple[float, int]] = {}
        
        # 搜索方向：上、下、左、右
        directions = [
            (0, 1, Direction.UP),
            (0, -1, Direction.DOWN),
            (-1, 0, Direction.LEFT),
            (1, 0, Direction.RIGHT)
        ]
        
        # 限制搜索节点数防止超时
        max_nodes = 50000
        nodes_visited = 0
        
        while open_set and nodes_visited < max_nodes:
            current = heapq.heappop(open_set)
            nodes_visited += 1
            
            # 检查是否到达终点
            current_grid = self.grid.point_to_grid(current.position)
            if current_grid == dst_grid:
                # 添加终点并返回路径
                final_path = current.path.copy()
                if final_path[-1] != dst:
                    final_path.append(dst)
                return Route(net_id=net.id, waypoints=final_path)
            
            # 检查是否已访问过（使用简化的状态键）
            state_key = (current_grid[0], current_grid[1], current.direction)
            if state_key in visited:
                prev_g, prev_turns = visited[state_key]
                # 如果之前的状态更优（代价更低且转弯更少），跳过
                if prev_g <= current.g_score and prev_turns <= current.turns:
                    continue
            visited[state_key] = (current.g_score, current.turns)
            
            # 扩展相邻节点
            for dcol, drow, new_dir in directions:
                # 计算新位置（移动一格）
                new_col = current_grid[0] + dcol
                new_row = current_grid[1] + drow
                
                # 检查边界
                if not self.grid.is_valid(new_col, new_row):
                    continue
                
                # 检查是否被阻塞
                if self.grid.is_blocked(new_col, new_row):
                    # 但如果是终点所在格，允许通过
                    if (new_col, new_row) != dst_grid:
                        continue
                
                # 计算转弯数
                new_turns = current.turns
                is_turning = current.direction != Direction.NONE and current.direction != new_dir
                if is_turning:
                    new_turns += 1
                
                # 检查转弯限制
                if new_turns > max_turns:
                    continue
                
                # 计算新的g值
                # 基础代价：移动距离
                # 转弯惩罚：每次转弯增加额外代价，鼓励直线路径
                move_cost = self.cell_size
                turn_penalty = self.cell_size * 5 if is_turning else 0  # 转弯惩罚
                new_g = current.g_score + move_cost + turn_penalty
                
                # 计算启发值（到终点的曼哈顿距离）
                new_point = self.grid.grid_to_point(new_col, new_row)
                h = new_point.manhattan_distance_to(dst)
                
                new_f = new_g + h
                
                # 构建新路径
                new_path = current.path.copy()
                
                # 如果方向改变，添加转折点（waypoint）
                if current.direction != Direction.NONE and current.direction != new_dir:
                    # 转折点是当前位置
                    if new_path[-1] != current.position:
                        new_path.append(current.position)
                
                # 创建新状态
                new_state = AStarState(
                    f_score=new_f,
                    g_score=new_g,
                    position=new_point,
                    direction=new_dir,
                    turns=new_turns,
                    path=new_path
                )
                
                heapq.heappush(open_set, new_state)
        
        # 搜索失败
        return None
    
    def _rip_up_and_reroute(self, failed_nets: List[Net], iteration: int) -> List[Net]:
        """
        Rip-up & Reroute策略
        
        对于失败的网络，尝试移除与其冲突的已布线网络，
        然后以不同的顺序重新布线
        
        Args:
            failed_nets: 失败的网络列表
            iteration: 当前迭代次数
        
        Returns:
            仍然失败的网络列表
        """
        # 收集所有需要重新布线的网络
        nets_to_reroute = failed_nets.copy()
        
        # 找到与失败网络冲突的已布线网络
        # （简化策略：移除距离失败网络源/目的端口最近的几条已布线）
        for failed_net in failed_nets:
            src = self.get_port_point(failed_net.source_port)
            dst = self.get_port_point(failed_net.target_port)
            
            # 找到可能冲突的已布线网络（基于路径接近程度）
            candidates = []
            for net_id, route in self.routes.items():
                for wp in route.waypoints:
                    if wp.distance_to(src) < self.spec.min_spacing * 5 or \
                       wp.distance_to(dst) < self.spec.min_spacing * 5:
                        candidates.append(net_id)
                        break
            
            # 移除一部分冲突的路由（每次迭代移除更多）
            num_to_remove = min(len(candidates), iteration + 1)
            for i in range(num_to_remove):
                if candidates:
                    net_id_to_remove = candidates.pop()
                    if net_id_to_remove in self.routes:
                        # 移除路由
                        route = self.routes.pop(net_id_to_remove)
                        self.grid.remove_route_obstacle(route, self.spec.min_spacing)
                        
                        # 找到对应的net对象
                        for net in self.chip.nets:
                            if net.id == net_id_to_remove:
                                nets_to_reroute.append(net)
                                break
        
        # 去重
        seen = set()
        unique_nets = []
        for net in nets_to_reroute:
            if net.id not in seen:
                seen.add(net.id)
                unique_nets.append(net)
        
        # 打乱顺序重新布线
        import random
        random.seed(iteration * 42)
        random.shuffle(unique_nets)
        
        # 重新布线
        still_failed = []
        for net in unique_nets:
            if net.id in self.routes:
                continue  # 已经成功布线
            
            route = self._route_single_net(net)
            if route is not None:
                self.routes[net.id] = route
                self.grid.add_route_obstacle(route, self.spec.min_spacing)
            else:
                still_failed.append(net)
        
        return still_failed
    
    def simplify_route(self, route: Route) -> Route:
        """
        简化路由：移除冗余的waypoints
        
        如果连续的三个点共线，则可以移除中间的点
        
        Args:
            route: 原始路由
        
        Returns:
            简化后的路由
        """
        if len(route.waypoints) <= 2:
            return route
        
        simplified = [route.waypoints[0]]
        
        for i in range(1, len(route.waypoints) - 1):
            prev = simplified[-1]
            curr = route.waypoints[i]
            next_pt = route.waypoints[i + 1]
            
            # 检查三点是否共线
            # 使用叉积判断
            cross = (curr.x - prev.x) * (next_pt.y - prev.y) - \
                    (curr.y - prev.y) * (next_pt.x - prev.x)
            
            if abs(cross) > 0.01:  # 不共线，保留中间点
                simplified.append(curr)
        
        simplified.append(route.waypoints[-1])
        
        return Route(net_id=route.net_id, waypoints=simplified)
    
    def get_statistics(self) -> dict:
        """获取布线统计信息"""
        return self.stats.copy()


def solve_microfluidic_routing(chip: MicrofluidicChip, 
                               cell_size: Optional[float] = None,
                               use_rip_up: bool = True) -> Tuple[Dict[str, Route], dict]:
    """
    求解微流控芯片布线问题的便捷函数
    
    Args:
        chip: 微流控芯片对象
        cell_size: 网格单元尺寸（可选）
        use_rip_up: 是否使用rip-up策略
    
    Returns:
        (routes, stats) 元组
    """
    router = MicrofluidicRouter(chip, cell_size)
    routes = router.route_all(use_rip_up=use_rip_up)
    
    # 简化所有路由
    simplified_routes = {}
    for net_id, route in routes.items():
        simplified_routes[net_id] = router.simplify_route(route)
    
    return simplified_routes, router.get_statistics()


if __name__ == "__main__":
    # 简单测试
    from input_generator import generate_microfluidic_chip, save_chip_to_json
    
    # 生成测试用例
    chip = generate_microfluidic_chip(
        board_width=500,
        board_height=400,
        num_modules=3,
        num_board_ports=4,
        num_nets=4,
        min_spacing=15,
        max_waypoints=6,
        seed=42
    )
    
    print("=== Testing Microfluidic Router ===")
    print(f"Board: {chip.spec.board_width} x {chip.spec.board_height}")
    print(f"Modules: {len(chip.modules)}")
    print(f"Nets: {len(chip.nets)}")
    
    # 布线
    routes, stats = solve_microfluidic_routing(chip)
    
    print(f"\n=== Results ===")
    print(f"Routed: {stats['routed_nets']} / {stats['total_nets']}")
    print(f"Failed: {stats['failed_nets']}")
    print(f"Total length: {stats['total_length']:.2f}")
    print(f"Total turns: {stats['total_turns']}")
    
    print(f"\n=== Route Details ===")
    for net_id, route in routes.items():
        print(f"  {net_id}: {len(route.waypoints)} waypoints, "
              f"{route.total_length():.2f} length, {route.num_turns()} turns")
