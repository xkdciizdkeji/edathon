"""
微流控芯片布线问题 - 可视化工具
Microfluidic Chip Routing Problem - Visualization Tool

提供芯片和布线结果的可视化功能：
- 绘制芯片板和模块
- 显示端口位置
- 绘制布线路径
- 输出PNG图像或显示交互窗口

作者: EDAthon 2025 参赛预备代码
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict, List, Optional, Tuple

from input_generator import MicrofluidicChip, Module, Port, Net
from router import Route, Point


# 配色方案
COLORS = {
    'board': '#f0f0f0',           # 芯片板背景色
    'module': '#4a90d9',          # 模块填充色
    'module_edge': '#2a70b9',     # 模块边框色
    'port': '#ff6b6b',            # 端口颜色
    'board_port': '#51cf66',      # 边缘端口颜色
    'route_colors': [             # 布线颜色（循环使用）
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
        '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
        '#16a085', '#c0392b', '#8e44ad', '#27ae60'
    ],
    'waypoint': '#000000',        # waypoint标记颜色
    'text': '#333333',            # 文字颜色
}


def visualize_chip(
    chip: MicrofluidicChip,
    routes: Optional[Dict[str, Route]] = None,
    title: str = "Microfluidic Chip",
    show_grid: bool = False,
    show_port_labels: bool = True,
    show_net_labels: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    可视化微流控芯片及其布线结果
    
    Args:
        chip: 微流控芯片对象
        routes: 布线结果（可选）
        title: 图表标题
        show_grid: 是否显示网格线
        show_port_labels: 是否显示端口标签
        show_net_labels: 是否显示网络标签
        figsize: 图表尺寸
        save_path: 保存路径（如果指定则保存图像）
    
    Returns:
        matplotlib Figure对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置坐标轴范围
    margin = 30
    ax.set_xlim(-margin, chip.spec.board_width + margin)
    ax.set_ylim(-margin, chip.spec.board_height + margin)
    ax.set_aspect('equal')
    
    # 绘制芯片板背景
    board_rect = patches.Rectangle(
        (0, 0), chip.spec.board_width, chip.spec.board_height,
        linewidth=2, edgecolor='#333333', facecolor=COLORS['board']
    )
    ax.add_patch(board_rect)
    
    # 显示网格线
    if show_grid:
        cell_size = chip.spec.min_spacing
        for x in np.arange(0, chip.spec.board_width, cell_size):
            ax.axvline(x, color='#dddddd', linewidth=0.5, alpha=0.5)
        for y in np.arange(0, chip.spec.board_height, cell_size):
            ax.axhline(y, color='#dddddd', linewidth=0.5, alpha=0.5)
    
    # 绘制模块
    for module in chip.modules:
        _draw_module(ax, module, show_port_labels)
    
    # 绘制边缘端口
    for port in chip.board_ports:
        _draw_port(ax, port, COLORS['board_port'], show_port_labels)
    
    # 绘制布线路径
    if routes:
        color_idx = 0
        for net_id, route in routes.items():
            color = COLORS['route_colors'][color_idx % len(COLORS['route_colors'])]
            _draw_route(ax, route, color, show_net_labels)
            color_idx += 1
    
    # 添加图例和标题
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (μm)', fontsize=10)
    ax.set_ylabel('Y (μm)', fontsize=10)
    
    # 添加信息框
    info_text = _create_info_text(chip, routes)
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def _draw_module(ax: plt.Axes, module: Module, show_labels: bool):
    """绘制单个模块"""
    # 模块矩形
    rect = patches.Rectangle(
        (module.x, module.y), module.width, module.height,
        linewidth=2,
        edgecolor=COLORS['module_edge'],
        facecolor=COLORS['module'],
        alpha=0.7
    )
    ax.add_patch(rect)
    
    # 模块标签
    ax.text(
        module.x + module.width / 2,
        module.y + module.height / 2,
        module.id,
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        color='white'
    )
    
    # 绘制模块端口
    for port in module.ports:
        _draw_port(ax, port, COLORS['port'], show_labels)


def _draw_port(ax: plt.Axes, port: Port, color: str, show_label: bool):
    """绘制端口"""
    # 端口圆点
    ax.plot(port.x, port.y, 'o', markersize=8, color=color, 
            markeredgecolor='white', markeredgewidth=1)
    
    # 端口标签
    if show_label:
        # 根据端口位置调整标签位置
        offset_x, offset_y = 5, 5
        ha, va = 'left', 'bottom'
        
        if port.x < 50:
            offset_x = -5
            ha = 'right'
        if port.y < 50:
            offset_y = -5
            va = 'top'
        
        ax.text(
            port.x + offset_x, port.y + offset_y,
            port.id.split('_')[-1],  # 简化标签
            fontsize=7, color=COLORS['text'],
            ha=ha, va=va, alpha=0.8
        )


def _draw_route(ax: plt.Axes, route: Route, color: str, show_label: bool):
    """绘制布线路径"""
    if len(route.waypoints) < 2:
        return
    
    # 准备线段数据
    segments = []
    for i in range(len(route.waypoints) - 1):
        p1 = route.waypoints[i]
        p2 = route.waypoints[i + 1]
        segments.append([(p1.x, p1.y), (p2.x, p2.y)])
    
    # 绘制路径线段
    lc = LineCollection(segments, colors=color, linewidths=2.5, alpha=0.8)
    ax.add_collection(lc)
    
    # 绘制waypoints（起点和终点除外）
    for i, wp in enumerate(route.waypoints):
        if i == 0 or i == len(route.waypoints) - 1:
            # 起点和终点用更大的标记
            ax.plot(wp.x, wp.y, 's', markersize=10, color=color,
                   markeredgecolor='white', markeredgewidth=1.5)
        else:
            # 中间waypoints用小圆点
            ax.plot(wp.x, wp.y, 'o', markersize=6, color=COLORS['waypoint'],
                   markeredgecolor='white', markeredgewidth=1)
    
    # 网络标签（在路径中点）
    if show_label and len(route.waypoints) >= 2:
        mid_idx = len(route.waypoints) // 2
        mid_point = route.waypoints[mid_idx]
        ax.text(
            mid_point.x, mid_point.y + 10,
            route.net_id,
            fontsize=8, color=color, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )


def _create_info_text(chip: MicrofluidicChip, routes: Optional[Dict[str, Route]]) -> str:
    """创建信息文本"""
    lines = [
        f"Board: {chip.spec.board_width:.0f} × {chip.spec.board_height:.0f}",
        f"Modules: {len(chip.modules)}",
        f"Nets: {len(chip.nets)}",
        f"Min Spacing: {chip.spec.min_spacing}",
        f"Max Waypoints: {chip.spec.max_waypoints}",
    ]
    
    if routes:
        routed = len(routes)
        total = len(chip.nets)
        success_rate = routed / total * 100 if total > 0 else 0
        
        total_length = sum(r.total_length() for r in routes.values())
        total_turns = sum(r.num_turns() for r in routes.values())
        
        lines.extend([
            "",
            f"Routed: {routed}/{total} ({success_rate:.1f}%)",
            f"Total Length: {total_length:.1f}",
            f"Total Turns: {total_turns}",
        ])
    
    return '\n'.join(lines)


def visualize_comparison(
    chip: MicrofluidicChip,
    routes_before: Optional[Dict[str, Route]],
    routes_after: Dict[str, Route],
    title: str = "Routing Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    并排比较两个布线结果
    
    Args:
        chip: 微流控芯片对象
        routes_before: 优化前的布线结果
        routes_after: 优化后的布线结果
        title: 图表标题
        save_path: 保存路径
    
    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 设置通用参数
    margin = 30
    
    for ax, routes, subtitle in zip(
        axes,
        [routes_before, routes_after],
        ["Before Optimization", "After Optimization"]
    ):
        ax.set_xlim(-margin, chip.spec.board_width + margin)
        ax.set_ylim(-margin, chip.spec.board_height + margin)
        ax.set_aspect('equal')
        
        # 绘制芯片板背景
        board_rect = patches.Rectangle(
            (0, 0), chip.spec.board_width, chip.spec.board_height,
            linewidth=2, edgecolor='#333333', facecolor=COLORS['board']
        )
        ax.add_patch(board_rect)
        
        # 绘制模块
        for module in chip.modules:
            _draw_module(ax, module, False)
        
        # 绘制边缘端口
        for port in chip.board_ports:
            _draw_port(ax, port, COLORS['board_port'], False)
        
        # 绘制布线路径
        if routes:
            color_idx = 0
            for net_id, route in routes.items():
                color = COLORS['route_colors'][color_idx % len(COLORS['route_colors'])]
                _draw_route(ax, route, color, False)
                color_idx += 1
        
        # 添加信息
        info_text = _create_info_text(chip, routes)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_title(subtitle, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (μm)', fontsize=10)
        ax.set_ylabel('Y (μm)', fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return fig


def create_animation(
    chip: MicrofluidicChip,
    routes: Dict[str, Route],
    save_path: str,
    fps: int = 2
):
    """
    创建布线过程的动画（需要安装imageio）
    
    Args:
        chip: 微流控芯片对象
        routes: 布线结果
        save_path: GIF保存路径
        fps: 帧率
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio")
        return
    
    images = []
    
    # 逐步添加路由
    partial_routes = {}
    for net_id, route in routes.items():
        partial_routes[net_id] = route
        
        fig = visualize_chip(
            chip, partial_routes,
            title=f"Routing Progress: {net_id}",
            show_port_labels=False,
            show_net_labels=False
        )
        
        # 保存到临时文件
        temp_path = f"_temp_frame_{len(images)}.png"
        plt.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        images.append(imageio.imread(temp_path))
        
        # 删除临时文件
        import os
        os.remove(temp_path)
    
    # 创建GIF
    imageio.mimsave(save_path, images, fps=fps)
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    # 测试可视化
    from input_generator import generate_microfluidic_chip
    from router import solve_microfluidic_routing
    import os
    
    # 生成测试用例
    chip = generate_microfluidic_chip(
        board_width=600,
        board_height=500,
        num_modules=4,
        num_board_ports=6,
        num_nets=6,
        min_spacing=20,
        max_waypoints=6,
        seed=42
    )
    
    # 布线
    routes, stats = solve_microfluidic_routing(chip)
    
    print("=== Visualization Test ===")
    print(f"Routed: {stats['routed_nets']} / {stats['total_nets']}")
    
    # 可视化
    output_dir = os.path.dirname(__file__)
    
    fig = visualize_chip(
        chip, routes,
        title="Microfluidic Chip Routing Result",
        show_grid=True,
        save_path=os.path.join(output_dir, "test_visualization.png")
    )
    
    plt.show()
