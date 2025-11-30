from collections import deque, defaultdict
from typing import Dict, List, Set, Optional, Union


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
                         dirty_nodes: Set[Union[int, str]]) -> tuple[Dict[Union[int, str], List[Union[int, str]]],
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
        raise ValueError(f"脏节点子图存在环，无法拓扑排序！已排序节点数：{len(top_order)}，总脏节点数：{len(in_degree)}")
    
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


# ------------------------------ 示例演示 ------------------------------
if __name__ == "__main__":
    # 示例1：简单DAG（节点为整数）
    print("=== 示例1：简单整数节点DAG ===")
    adj1 = {
        "A": ["B", "E"],
        "B": ["C"],
        "C": ["D"],
        "E": [],
        "D": [],
        "F": ["G"],  # 非脏节点（无源头指向，不会被标记）
        "G": []
    }
    dirty_sources1 = ["B", "E"]  # 脏节点源头：B和E（B的后代C、D也为脏节点）
    result1 = dirty_nodes_top_sort(adj1, dirty_sources1)
    print(f"原邻接表：{adj1}")
    print(f"脏节点源头：{dirty_sources1}")
    print(f"脏节点拓扑序：{result1}")  # 可能输出：['B', 'E', 'C', 'D'] 或 ['E', 'B', 'C', 'D']（顺序不唯一，均合法）
    
    # 示例2：含孤立节点的DAG（脏节点源头为孤立节点）
    print("\n=== 示例2：含孤立节点的DAG ===")
    adj2 = {
        1: [2],
        2: [3],
        3: [],
        4: [],  # 孤立节点
        5: [6]
    }
    dirty_sources2 = [4, 5]  # 4是孤立节点（无后代），5的后代是6
    result2 = dirty_nodes_top_sort(adj2, dirty_sources2)
    print(f"原邻接表：{adj2}")
    print(f"脏节点源头：{dirty_sources2}")
    print(f"脏节点拓扑序：{result2}")  # 输出：[4,5,6] 或 [5,4,6]
    
    # 示例3：子图存在环（测试环检测）
    print("\n=== 示例3：子图存在环 ===")
    adj3 = {
        "X": ["Y"],
        "Y": ["Z"],
        "Z": ["Y"],  # Y和Z形成环
        "W": []
    }
    dirty_sources3 = ["X"]  # 脏节点：X、Y、Z（子图有环）
    result3 = dirty_nodes_top_sort(adj3, dirty_sources3)
    print(f"原邻接表：{adj3}")
    print(f"脏节点源头：{dirty_sources3}")
    print(f"脏节点拓扑序：{result3}")  # 输出错误提示，返回空列表
    
    # 示例4：多DAG处理（按DAG独立调用即可）
    print("\n=== 示例4：多DAG处理 ===")
    # DAG1
    adj4_1 = {"a": ["b"], "b": ["c"], "c": []}
    dirty_sources4_1 = ["a"]
    result4_1 = dirty_nodes_top_sort(adj4_1, dirty_sources4_1)
    # DAG2
    adj4_2 = {"x": ["y"], "y": [], "z": []}
    dirty_sources4_2 = ["z", "x"]
    result4_2 = dirty_nodes_top_sort(adj4_2, dirty_sources4_2)
    # 整合多DAG结果
    multi_dag_result = {
        "DAG1": result4_1,
        "DAG2": result4_2
    }
    print(f"DAG1拓扑序：{result4_1}")  # 输出：['a', 'b', 'c']
    print(f"DAG2拓扑序：{result4_2}")  # 输出：['z', 'x', 'y'] 或 ['x', 'z', 'y']
    print(f"多DAG整合结果：{multi_dag_result}")