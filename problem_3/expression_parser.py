"""
表达式解析器：解析表达式并构建依赖图（DAG）

主要功能：
1. 解析表达式字符串，提取变量依赖关系
2. 构建依赖图（DAG）
3. 检测循环依赖（SCC）
4. 提取独立的 DAG 子图
"""

import re
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Union


class ExpressionParser:
    """
    表达式解析器
    
    负责解析表达式并提取变量依赖关系
    支持的表达式格式：
    - 算术运算：+, -, *, /, %, **
    - 数学函数：sin, cos, tan, sqrt, abs, log, exp
    - 三元表达式：condition ? true_val : false_val
    - 比较运算：>, <, >=, <=, ==, !=
    """
    
    # 匹配变量名的正则表达式（字母或下划线开头，后跟字母数字下划线）
    VAR_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    
    # 保留关键字（这些不是变量名）
    RESERVED_KEYWORDS = {
        'sin', 'cos', 'tan', 'sqrt', 'abs', 'log', 'exp', 'pow',
        'min', 'max', 'floor', 'ceil', 'round',
        'true', 'false', 'True', 'False'
    }
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    def extract_dependencies(self, expression: str, 
                            all_vars: Set[str]) -> Set[str]:
        """
        从表达式中提取依赖的变量
        
        :param expression: 表达式字符串
        :param all_vars: 所有已定义的变量名集合
        :return: 该表达式依赖的变量名集合
        """
        # 找出表达式中所有可能的变量名
        matches = self.VAR_PATTERN.findall(expression)
        
        # 过滤：只保留实际存在的变量（排除关键字和未定义的标识符）
        dependencies = set()
        for name in matches:
            if name in all_vars and name not in self.RESERVED_KEYWORDS:
                dependencies.add(name)
        
        return dependencies
    
    def parse_expressions(self, expressions: Dict[str, str]) -> Dict[str, Set[str]]:
        """
        解析所有表达式，提取每个变量的依赖关系
        
        :param expressions: 变量名 -> 表达式字符串 的字典
        :return: 变量名 -> 依赖变量集合 的字典
        """
        all_vars = set(expressions.keys())
        dependencies = {}
        
        for var_name, expr in expressions.items():
            # 提取该表达式依赖的变量（排除自身）
            deps = self.extract_dependencies(expr, all_vars)
            deps.discard(var_name)  # 移除自身依赖（如果有的话）
            dependencies[var_name] = deps
        
        return dependencies


class DependencyGraph:
    """
    依赖图（DAG）管理器
    
    维护表达式之间的依赖关系，支持：
    1. 构建正向图（被依赖 -> 依赖者）和反向图（依赖者 -> 被依赖）
    2. 拓扑排序
    3. 检测循环依赖（SCC）
    4. 提取独立子图
    5. 脏节点标记与传播
    """
    
    def __init__(self):
        """初始化依赖图"""
        # 正向图：adj_out[A] = [B, C] 表示 B 和 C 依赖于 A
        # 即 A 改变时，需要重新计算 B 和 C
        self.adj_out: Dict[str, List[str]] = defaultdict(list)
        
        # 反向图：adj_in[B] = [A] 表示 B 依赖于 A
        # 用于拓扑排序和计算顺序
        self.adj_in: Dict[str, List[str]] = defaultdict(list)
        
        # 所有节点集合
        self.nodes: Set[str] = set()
        
        # 基础节点（叶节点，无依赖）
        self.base_nodes: Set[str] = set()
        
        # 缓存的拓扑排序结果
        self._topo_order: Optional[List[str]] = None
    
    def build_from_dependencies(self, dependencies: Dict[str, Set[str]]):
        """
        根据依赖关系构建图
        
        :param dependencies: 变量名 -> 依赖变量集合 的字典
        """
        self.adj_out.clear()
        self.adj_in.clear()
        self.nodes.clear()
        self.base_nodes.clear()
        self._topo_order = None
        
        # 收集所有节点
        for var_name, deps in dependencies.items():
            self.nodes.add(var_name)
            self.nodes.update(deps)
        
        # 构建邻接表
        for var_name, deps in dependencies.items():
            for dep in deps:
                # dep -> var_name（var_name 依赖于 dep）
                self.adj_out[dep].append(var_name)
                self.adj_in[var_name].append(dep)
        
        # 识别基础节点（入度为0的节点）
        for node in self.nodes:
            if len(self.adj_in[node]) == 0:
                self.base_nodes.add(node)
    
    def get_topological_order(self) -> List[str]:
        """
        获取拓扑排序结果（使用 Kahn 算法）
        
        拓扑排序保证：如果 A 依赖于 B，则 B 在序列中出现在 A 之前
        这样按顺序计算可以确保每个变量的依赖都已计算完毕
        
        :return: 拓扑排序后的节点列表
        :raises ValueError: 如果存在循环依赖
        """
        if self._topo_order is not None:
            return self._topo_order
        
        # 计算每个节点的入度
        in_degree = {node: len(self.adj_in[node]) for node in self.nodes}
        
        # 初始化队列：入度为0的节点
        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            # 更新后继节点的入度
            for succ in self.adj_out[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        
        # 检测环：如果拓扑序列长度小于节点数，说明存在环
        if len(topo_order) != len(self.nodes):
            remaining = [n for n in self.nodes if n not in set(topo_order)]
            raise ValueError(f"检测到循环依赖！涉及节点: {remaining}")
        
        self._topo_order = topo_order
        return topo_order
    
    def detect_cycles(self) -> List[List[str]]:
        """
        检测图中的强连通分量（SCC），即循环依赖
        
        使用 Tarjan 算法
        
        :return: 所有包含多于一个节点的 SCC 列表（即循环）
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for succ in self.adj_out[node]:
                if succ not in index:
                    strongconnect(succ)
                    lowlinks[node] = min(lowlinks[node], lowlinks[succ])
                elif on_stack.get(succ, False):
                    lowlinks[node] = min(lowlinks[node], index[succ])
            
            if lowlinks[node] == index[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                sccs.append(scc)
        
        for node in self.nodes:
            if node not in index:
                strongconnect(node)
        
        # 只返回大小大于1的 SCC（即真正的循环）
        return [scc for scc in sccs if len(scc) > 1]
    
    def extract_independent_dags(self) -> List['DependencyGraph']:
        """
        提取所有独立的 DAG 子图（基于弱连通分量）
        
        :return: 独立 DAG 列表
        """
        if not self.nodes:
            return []
        
        visited = set()
        independent_dags = []
        
        for start_node in self.nodes:
            if start_node not in visited:
                # BFS 找出弱连通分量
                component = set()
                queue = deque([start_node])
                visited.add(start_node)
                component.add(start_node)
                
                while queue:
                    current = queue.popleft()
                    # 正向遍历
                    for succ in self.adj_out[current]:
                        if succ not in visited:
                            visited.add(succ)
                            component.add(succ)
                            queue.append(succ)
                    # 反向遍历
                    for pred in self.adj_in[current]:
                        if pred not in visited:
                            visited.add(pred)
                            component.add(pred)
                            queue.append(pred)
                
                # 为该分量创建子图
                sub_graph = DependencyGraph()
                sub_graph.nodes = component
                for node in component:
                    sub_graph.adj_out[node] = [s for s in self.adj_out[node] if s in component]
                    sub_graph.adj_in[node] = [p for p in self.adj_in[node] if p in component]
                    if len(sub_graph.adj_in[node]) == 0:
                        sub_graph.base_nodes.add(node)
                
                independent_dags.append(sub_graph)
        
        return independent_dags


class DirtyNodeTracker:
    """
    脏节点追踪器
    
    当某些节点的值发生变化时，追踪并标记所有需要重新计算的节点
    
    核心算法：
    1. 从变化的源节点出发
    2. 通过 BFS/DFS 沿正向边（adj_out）传播
    3. 标记所有可达节点为"脏"
    4. 对脏节点进行拓扑排序，得到正确的重算顺序
    """
    
    def __init__(self, graph: DependencyGraph):
        """
        初始化追踪器
        
        :param graph: 依赖图
        """
        self.graph = graph
    
    def mark_dirty_nodes(self, changed_sources: List[str]) -> Set[str]:
        """
        标记所有脏节点
        
        脏节点 = 变化源节点 + 所有直接或间接依赖于变化源的节点
        
        :param changed_sources: 发生变化的源节点列表
        :return: 所有脏节点的集合
        """
        dirty_nodes = set()
        queue = deque()
        
        # 初始化：将所有变化的源节点加入队列
        for node in changed_sources:
            if node in self.graph.nodes:
                dirty_nodes.add(node)
                queue.append(node)
        
        # BFS 传播脏标记
        while queue:
            current = queue.popleft()
            # 遍历所有依赖于 current 的节点
            for succ in self.graph.adj_out[current]:
                if succ not in dirty_nodes:
                    dirty_nodes.add(succ)
                    queue.append(succ)
        
        return dirty_nodes
    
    def get_dirty_subgraph_topo_order(self, dirty_nodes: Set[str]) -> List[str]:
        """
        对脏节点子图进行拓扑排序
        
        只考虑脏节点之间的边，返回正确的重算顺序
        
        :param dirty_nodes: 脏节点集合
        :return: 脏节点的拓扑排序结果
        """
        if not dirty_nodes:
            return []
        
        # 构建子图的入度表
        # 注意：只计算来自脏节点的入边
        in_degree = {}
        for node in dirty_nodes:
            count = 0
            for pred in self.graph.adj_in[node]:
                if pred in dirty_nodes:
                    count += 1
            in_degree[node] = count
        
        # Kahn 算法拓扑排序
        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            for succ in self.graph.adj_out[node]:
                if succ in dirty_nodes:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)
        
        # 验证排序完整性
        if len(topo_order) != len(dirty_nodes):
            raise ValueError("脏节点子图存在循环依赖！")
        
        return topo_order
    
    def get_recompute_order(self, changed_sources: List[str]) -> Tuple[Set[str], List[str]]:
        """
        获取需要重新计算的节点及其计算顺序
        
        这是对外的主要接口，整合了脏标记和拓扑排序
        
        :param changed_sources: 发生变化的源节点列表
        :return: (脏节点集合, 重算顺序列表)
        """
        # 1. 标记脏节点
        dirty_nodes = self.mark_dirty_nodes(changed_sources)
        
        # 2. 对脏节点进行拓扑排序
        topo_order = self.get_dirty_subgraph_topo_order(dirty_nodes)
        
        return dirty_nodes, topo_order


def build_dependency_graph(expressions: Dict[str, str]) -> DependencyGraph:
    """
    便捷函数：从表达式字典构建依赖图
    
    :param expressions: 变量名 -> 表达式字符串 的字典
    :return: 构建好的依赖图
    """
    parser = ExpressionParser()
    dependencies = parser.parse_expressions(expressions)
    
    graph = DependencyGraph()
    graph.build_from_dependencies(dependencies)
    
    return graph


if __name__ == "__main__":
    # 测试示例
    print("=" * 50)
    print("表达式解析器测试")
    print("=" * 50)
    
    # 题目示例
    expressions = {
        "var1": "5.0",
        "A": "var1 + 2",
        "B": "A / 2",
        "C": "(B > 1.0) ? var1 : A"
    }
    
    print("\n输入表达式:")
    for var, expr in expressions.items():
        print(f"  {var} = {expr}")
    
    # 构建依赖图
    graph = build_dependency_graph(expressions)
    
    print("\n依赖关系 (adj_in):")
    for node in graph.nodes:
        deps = graph.adj_in[node]
        if deps:
            print(f"  {node} 依赖于: {deps}")
        else:
            print(f"  {node} (基础节点)")
    
    print("\n影响关系 (adj_out):")
    for node in graph.nodes:
        affected = graph.adj_out[node]
        if affected:
            print(f"  {node} 影响: {affected}")
    
    print("\n拓扑排序:")
    topo = graph.get_topological_order()
    print(f"  {' -> '.join(topo)}")
    
    # 测试脏节点追踪
    print("\n脏节点追踪测试:")
    tracker = DirtyNodeTracker(graph)
    
    changed = ["var1"]
    dirty, order = tracker.get_recompute_order(changed)
    print(f"  变化源: {changed}")
    print(f"  脏节点: {dirty}")
    print(f"  重算顺序: {order}")
