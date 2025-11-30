from collections import deque, defaultdict
from typing import Dict, List, Set, Union, Tuple

# ------------------------------ 核心函数：从混合邻接表提取所有独立DAG ------------------------------
def extract_independent_dags(mixed_adj: Dict[Union[int, str], List[Union[int, str]]]) -> List[Dict[Union[int, str], List[Union[int, str]]]]:
    """
    从混合邻接表中自动提取所有独立DAG（基于弱连通分量检测）
    :param mixed_adj: 混合邻接表（包含多个无关联的独立DAG）
    :return: 独立DAG的列表（每个元素是一个DAG的邻接表）
    """
    if not mixed_adj:
        return []
    
    # 步骤1：收集所有节点（避免遗漏“仅作为边的终点，不在邻接表key中”的节点）
    all_nodes = set()
    # 先加入邻接表的key（有出边的节点）
    for node in mixed_adj.keys():
        all_nodes.add(node)
    # 再加入所有边的终点（可能无出边，不在key中）
    for successors in mixed_adj.values():
        for succ in successors:
            all_nodes.add(succ)
    all_nodes = list(all_nodes)  # 转为列表便于遍历
    
    # 步骤2：构建反向邻接表（存储每个节点的前驱，用于弱连通分量检测）
    reverse_adj = defaultdict(list)
    for u, successors in mixed_adj.items():
        for v in successors:
            reverse_adj[v].append(u)  # v的前驱是u，记录反向边
    
    # 步骤3：BFS检测弱连通分量（每个分量就是一个独立DAG）
    visited = set()
    independent_dags = []
    
    for start_node in all_nodes:
        if start_node not in visited:
            # BFS遍历弱连通分量（同时走正向和反向边，忽略方向）
            component = set()
            q = deque([start_node])
            visited.add(start_node)
            component.add(start_node)
            
            while q:
                current = q.popleft()
                # 遍历正向边的后继（u→v）
                for succ in mixed_adj.get(current, []):
                    if succ not in visited:
                        visited.add(succ)
                        component.add(succ)
                        q.append(succ)
                # 遍历反向边的前驱（v→u，即u是v的前驱）
                for pred in reverse_adj.get(current, []):
                    if pred not in visited:
                        visited.add(pred)
                        component.add(pred)
                        q.append(pred)
            
            # 步骤4：为当前分量构建独立DAG的邻接表（仅保留分量内的边）
            dag_adj = defaultdict(list)
            for u in component:
                # 遍历原邻接表中u的后继，仅保留也在分量中的节点
                for v in mixed_adj.get(u, []):
                    if v in component:
                        dag_adj[u].append(v)
            # 转为普通字典，避免defaultdict副作用
            independent_dags.append(dict(dag_adj))
    
    return independent_dags


# ------------------------------ 复用之前的脏节点拓扑排序代码 ------------------------------
def mark_dirty_nodes(adj: Dict[Union[int, str], List[Union[int, str]]],
                     dirty_sources: List[Union[int, str]]) -> Set[Union[int, str]]:
    """
    第一步：BFS标记所有脏节点（脏节点源头 + 其所有后代节点）
    :param adj: 原DAG的邻接表（key: 节点, value: 后继节点列表）
    :param dirty_sources: 脏节点源头列表
    :return: 所有脏节点的集合
    """
    dirty_nodes = set()
    if not dirty_sources:
        return dirty_nodes
    
    # 初始化队列，加入所有脏节点源头（过滤掉不在邻接表中的孤立节点，仍视为脏节点）
    q = deque()
    for node in dirty_sources:
        if node not in dirty_nodes:
            dirty_nodes.add(node)
            q.append(node)
    
    # BFS正向遍历，收集所有后代节点
    while q:
        current = q.popleft()
        # 处理孤立节点（无后继）
        successors = adj.get(current, [])
        for succ in successors:
            if succ not in dirty_nodes:
                dirty_nodes.add(succ)
                q.append(succ)
    
    return dirty_nodes


def build_dirty_subgraph(adj: Dict[Union[int, str], List[Union[int, str]]],
                         dirty_nodes: Set[Union[int, str]]) -> Tuple[Dict[Union[int, str], List[Union[int, str]]],
                                                                     Dict[Union[int, str], int]]:
    """
    第二步：构建脏节点子图（仅包含脏节点及它们之间的边）
    :param adj: 原DAG的邻接表
    :param dirty_nodes: 所有脏节点的集合
    :return: 子图邻接表 + 子图节点入度字典
    """
    sub_adj = defaultdict(list)  # 子图邻接表
    in_degree = defaultdict(int)  # 子图节点入度
    
    # 初始化所有脏节点的入度为0（避免遗漏孤立脏节点）
    for node in dirty_nodes:
        in_degree[node] = 0
    
    # 遍历所有脏节点，构建子图边集和入度
    for u in dirty_nodes:
        # 原节点的所有后继
        successors = adj.get(u, [])
        for v in successors:
            # 仅保留双方都是脏节点的边
            if v in dirty_nodes:
                sub_adj[u].append(v)
                in_degree[v] += 1
    return dict(sub_adj), dict(in_degree)


def kahn_top_sort(sub_adj: Dict[Union[int, str], List[Union[int, str]]],
                  in_degree: Dict[Union[int, str], int]) -> List[Union[int, str]]:
    """
    第三步：Kahn算法对脏节点子图进行拓扑排序
    :param sub_adj: 脏节点子图的邻接表
    :param in_degree: 脏节点子图的入度字典
    :return: 脏节点的拓扑序列表
    :raises ValueError: 子图存在环时抛出异常
    """
    top_order = []
    # 初始化队列：入度为0的脏节点
    q = deque([node for node, degree in in_degree.items() if degree == 0])
    while q:
        current = q.popleft()
        top_order.append(current)
        
        # 遍历当前节点的后继，更新入度
        for succ in sub_adj.get(current, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                q.append(succ)
    
    # 环检测：若拓扑序长度 != 脏节点数，说明存在环
    if len(top_order) != len(in_degree):
        raise ValueError(f"子图存在环，无法拓扑排序！已排序节点数：{len(top_order)}，总脏节点数：{len(in_degree)}")
    return top_order


def dirty_nodes_top_sort(adj: Dict[Union[int, str], List[Union[int, str]]],
                         dirty_sources: List[Union[int, str]]) -> List[Union[int, str]]:
    """
    主函数：整合流程，实现脏节点的拓扑排序
    :param adj: 原DAG的邻接表（key: 节点, value: 后继节点列表）
    :param dirty_sources: 脏节点源头列表（后续所有后代均为脏节点）
    :return: 脏节点的拓扑序
    """
    # 1. 标记所有脏节点（源头 + 所有后代）
    dirty_nodes = mark_dirty_nodes(adj, dirty_sources)
    if not dirty_nodes:
        print("警告：无有效脏节点，返回空列表")
        return []
    
    # 2. 构建脏节点子图
    sub_adj, sub_in_degree = build_dirty_subgraph(adj, dirty_nodes)
    
    # 3. Kahn算法拓扑排序
    try:
        top_order = kahn_top_sort(sub_adj, sub_in_degree)
    except ValueError as e:
        print(f"错误：{e}")
        return []
    return top_order


# ------------------------------ 整合流程：提取独立DAG + 批量处理脏节点排序 ------------------------------
def process_mixed_dag(mixed_adj: Dict[Union[int, str], List[Union[int, str]]],
                      dirty_sources: List[Union[int, str]]) -> Dict[int, List[Union[int, str]]]:
    """
    处理混合邻接表：提取独立DAG → 逐个处理脏节点拓扑排序
    :param mixed_adj: 混合邻接表（含多个独立DAG）
    :param dirty_sources: 所有脏节点源头（可能分布在不同DAG中）
    :return: 键为DAG索引，值为该DAG的脏节点拓扑序
    """
    # 步骤1：提取所有独立DAG
    independent_dags = extract_independent_dags(mixed_adj)
    if not independent_dags:
        print("警告：未提取到任何独立DAG")
        return {}
    
    # 步骤2：为每个独立DAG筛选专属脏节点源头（仅保留该DAG中的节点）
    result = {}
    for dag_idx, dag_adj in enumerate(independent_dags):
        # 该DAG的所有节点（邻接表key + 所有边的终点）
        dag_nodes = set(dag_adj.keys())
        for succ_list in dag_adj.values():
            dag_nodes.update(succ_list)
        # 筛选出属于当前DAG的脏节点源头
        dag_dirty_sources = [node for node in dirty_sources if node in dag_nodes]
        # 处理当前DAG的脏节点排序
        dag_top_order = dirty_nodes_top_sort(dag_adj, dag_dirty_sources)
        result[dag_idx] = dag_top_order
    
    return result


# ------------------------------ 示例演示 ------------------------------
if __name__ == "__main__":
    # 示例：混合邻接表（包含4个独立DAG）
    # DAG1: A→B→C
    # DAG2: D→E（E不在邻接表key中，仅作为边的终点）
    # DAG3: F（孤立节点，无入边无出边）
    # DAG4: G→H→I
    mixed_adj = {
        "A": ["B"],
        "B": ["C"],
        "D": ["E"],  # E不在key中
        "G": ["H"],
        "H": ["I"]
        # F不在key中，是孤立节点
    }
    # 脏节点源头：分布在不同DAG中（A属于DAG1，D属于DAG2，F属于DAG3，I属于DAG4）
    dirty_sources = ["A", "D", "F", "I"]
    
    print("=== 输入混合邻接表 ===")
    print(mixed_adj)
    print("=== 脏节点源头 ===")
    print(dirty_sources)
    
    # 步骤1：提取独立DAG
    independent_dags = extract_independent_dags(mixed_adj)
    print("\n=== 提取的独立DAG ===")
    for i, dag in enumerate(independent_dags):
        print(f"DAG{i}：{dag}")  # 输出4个独立DAG，包括F的邻接表（空字典）
    
    # 步骤2：批量处理脏节点排序
    result = process_mixed_dag(mixed_adj, dirty_sources)
    print("\n=== 各DAG的脏节点拓扑序 ===")
    for dag_idx, top_order in result.items():
        print(f"DAG{dag_idx}：{top_order}")
    # 预期输出：
    # DAG0（A→B→C）：脏源头A → 脏节点A、B、C → 拓扑序['A','B','C']
    # DAG1（D→E）：脏源头D → 脏节点D、E → 拓扑序['D','E']
    # DAG2（F）：脏源头F → 脏节点F → 拓扑序['F']
    # DAG3（G→H→I）：脏源头I → 脏节点I（无后代） → 拓扑序['I']