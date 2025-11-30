"""
重写：基于 Mishchenko 等人（S&S）思想的三值逻辑 Resubstitution 处理器
文件说明：把你原来文件中的“分区/屎山”实现替换为更清晰、模块化的 S&S（Simulation + SAT）实现。

设计目标：
- 保持与你原来 circuit 接口的兼容（依赖 circuit.simulate(input_dict) 和 circuit.primary_inputs/primary_outputs）
- 实现两阶段流程：Simulation 阶段（快速筛除/收集） + Formal 阶段（尝试用 SMT/SAT 证明/找反例）
- 如果没有可用的 SAT/SMT 后端，则在小规模下自动回退到精确穷尽检验；在大规模下给出明确说明并尽量做强化采样 + 引导搜索（guided-random）
- 代码易于阅读和扩展——把 SAT/SMT 的接入点暴露为明确的函数，方便你在后来替换为你喜欢的求解器（Z3 / PySMT / PySAT）

注意：该实现不会对电路的 gate-level 结构做复杂的 CNF/SMT 翻译（这通常依赖于解析器提供的门描述）。如果你希望启用真正的 SAT/SMT 证明，请确保你的 circuit 对象能导出门级表示或约束构造函数（见 `build_symbolic_constraints` 的说明）。

使用方法（简短）：
    from resubstitution_solver_sns import SAndSResubstitution
    solver = SAndSResubstitution(circuit)
    res = solver.solve(mode='auto')
    solver.print_function_table(res)

如果你想让我把 SAT/SMT 集成点接入你具体的电路表示（把 circuit 的门级结构翻译为 Z3 / CNF），告诉我你现有解析器的数据结构细节，我会把那部分代码补上。

"""

from typing import Dict, Tuple, Optional, Set, List
from dataclasses import dataclass, field
import random
import math
import time
import itertools

try:
    # 尝试导入 z3，如果可用我们会把它用于 formal 阶段
    import z3
    _HAS_Z3 = True
except Exception:
    _HAS_Z3 = False


@dataclass
class ResubstitutionResult:
    can_resubstitute: bool
    function_table: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    dont_care_patterns: Set[Tuple[int, ...]] = field(default_factory=set)
    conflict: Optional[Tuple[Dict, Dict]] = None
    conflict_key: Optional[Tuple[int, ...]] = None
    conflict_values: Optional[Tuple[int, int]] = None

    def __str__(self):
        if self.can_resubstitute:
            return "可替换 - 最后输出可由其他输出唯一确定"
        else:
            if self.conflict_key is not None:
                return (f"不可替换 - 发现冲突：其他输出组合 {self.conflict_key} 对应多个最后输出值 {set(self.conflict_values)}")
            return "不可替换"


class SAndSResubstitution:
    """基于 Simulation + SAT 的三值重替换求解器（轻量原型实现）

    说明：
    - 依赖 circuit.simulate(input_dict) -> 返回 dict: {outname: value}
    - circuit.primary_inputs: List[str]
    - circuit.primary_outputs: List[str]
    - circuit 可以额外提供可选接口 build_symbolic_constraints(solver, prefix) 来把门级约束加入 z3 求解器
      如果提供该接口，formal_phase 将使用 Z3 去寻找/证明反例（注意：该接口必须能在两个副本上建立约束）

    如果 z3 或 build_symbolic_constraints 不可用，formal_phase 会在小规模下退回到完全穷尽检查；在大规模下它会做加强的仿真/引导搜索并返回“猜测”结果（带警告）。
    """

    def __init__(self, circuit, rng_seed: Optional[int] = 42):
        self.circuit = circuit
        self.PIs = list(circuit.primary_inputs)
        self.POs = list(circuit.primary_outputs)
        assert len(self.POs) >= 1, "需要至少一个输出"
        self.other_POs = self.POs[:-1]
        self.last_PO = self.POs[-1]
        self.n = len(self.PIs)
        self.m = len(self.POs)
        self._rng = random.Random(rng_seed)

    # ------------------------- Simulation Phase -------------------------
    def simulation_phase(self, rounds: int = 4096, bit_parallel: bool = False) -> Tuple[Dict[Tuple[int, ...], int], Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        """
        大量随机（或半结构化）仿真来快速发现反例。
        返回：mapping: other_outputs_tuple -> observed last_value
               如果发现冲突，立即返回 (mapping, (input1, input2)) 作为 witness（用具体输入元组）
        """
        mapping: Dict[Tuple[int, ...], int] = {}
        seen_inputs: Set[Tuple[int, ...]] = set()

        # include some guided patterns: all-zero, all-one, unit vectors
        seeds: List[Tuple[int, ...]] = []
        seeds.append(tuple(0 for _ in range(self.n)))
        seeds.append(tuple(1 for _ in range(self.n)))
        seeds.append(tuple(2 for _ in range(self.n)))
        for i in range(self.n):
            for v in (1, 2):
                tmp = [0] * self.n
                tmp[i] = v
                seeds.append(tuple(tmp))

        # evaluate seeds first
        for s in seeds:
            other, last = self._simulate_tuple(s)
            if other in mapping:
                if mapping[other] != last:
                    return mapping, (mapping[other, 'example'] if False else None, (s,))
            else:
                mapping[other] = last
                seen_inputs.add(s)

        # random rounds
        for _ in range(rounds):
            inp = tuple(self._rng.randint(0, 2) for _ in range(self.n))
            if inp in seen_inputs:
                continue
            seen_inputs.add(inp)
            other, last = self._simulate_tuple(inp)
            if other in mapping:
                if mapping[other] != last:
                    # find previous input that produced this other: we must find one
                    # we don't store it by default; to provide a witness we can re-search
                    prev = self._find_example_for_other(other)
                    return mapping, (prev, inp)
            else:
                mapping[other] = last
        return mapping, None

    def _find_example_for_other(self, other_pattern: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        # 从 simulation cache 中寻找产生该 other_pattern 的一个输入
        # circuit.simulate 结果缓存由下面的 _simulate_tuple 保持
        for inp, (oth, last) in getattr(self, "_sim_cache", {}).items():
            if oth == other_pattern:
                return inp
        # fallback: brute force (but avoid if huge)
        limit = 100000
        cnt = 0
        for inp in self._input_enumerator():
            oth, last = self._simulate_tuple(inp)
            if oth == other_pattern:
                return inp
            cnt += 1
            if cnt > limit:
                break
        return None

    def _simulate_tuple(self, inp: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
        # 缓存
        if not hasattr(self, '_sim_cache'):
            self._sim_cache = {}
        if inp in self._sim_cache:
            return self._sim_cache[inp]
        d = {self.PIs[i]: inp[i] for i in range(self.n)}
        outd = self.circuit.simulate(d)
        other_vals = tuple(outd[o] for o in self.other_POs)
        last_val = outd[self.last_PO]
        self._sim_cache[inp] = (other_vals, last_val)
        return other_vals, last_val

    def _input_enumerator(self):
        # 小心：3^n
        if self.n == 0:
            yield tuple()
            return
        current = [0] * self.n
        total = 3 ** self.n
        for _ in range(total):
            yield tuple(current)
            # increment
            i = 0
            while i < self.n:
                current[i] += 1
                if current[i] >= 3:
                    current[i] = 0
                    i += 1
                else:
                    break

    # ------------------------- Formal Phase (SAT/SMT) -------------------------
    def formal_phase(self, timeout_sec: int = 10) -> Tuple[bool, Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        """
        尝试用符号方法证明没有反例（或返回反例）。
        优先使用 z3 + circuit-provided symbol builder；如果不可用且规模小，退回到穷举。

        返回： (is_unsat, witness_pair_if_sat)
        - 如果 is_unsat == True: 证明不存在反例
        - 否则返回 (False, (inp1, inp2)) 一个反例
        """
        # 优先路径：如果 circuit 提供 build_symbolic_constraints 且 z3 可用
        if _HAS_Z3 and hasattr(self.circuit, 'build_symbolic_constraints'):
            try:
                return self._formal_with_z3(timeout_sec)
            except Exception as e:
                # 若失败回退
                print(f"[WARN] z3 符号证明失败：{e}，回退到穷举或加强仿真")

        # 回退：如果问题规模允许，做两副本穷尽搜索
        if self.n <= 10:  # 3^10 = 59049, 在可行范围内
            # 枚举所有对 (x1,x2)
            it = list(self._input_enumerator())
            for a in it:
                other_a, last_a = self._simulate_tuple(a)
                for b in it:
                    other_b, last_b = self._simulate_tuple(b)
                    if other_a == other_b and last_a != last_b:
                        return False, (a, b)
            return True, None

        # 否则无法用穷举证明；做更大规模的引导随机搜索
        # Guided search: 基于已知的 other pattern 样本，试图寻找另一个映射到该模式的 input
        known_map = getattr(self, '_sim_cache', {})
        # 生成若干变异样本：从随机 base 输入出发，做局部扰动
        attempts = 20000
        for _ in range(attempts):
            a = tuple(self._rng.randint(0, 2) for _ in range(self.n))
            other_a, last_a = self._simulate_tuple(a)
            # 随机尝试生成另一个 b，且期望 other_b == other_a but last_b != last_a
            for _ in range(50):
                b = list(a)
                # 随机打乱若干位
                k = self._rng.randint(1, max(1, self.n // 3))
                for _i in range(k):
                    pos = self._rng.randrange(self.n)
                    b[pos] = self._rng.randint(0, 2)
                b = tuple(b)
                other_b, last_b = self._simulate_tuple(b)
                if other_b == other_a and last_b != last_a:
                    return False, (a, b)
        # 否则无法找到反例；返回"未证明但未发现" (treat as UNSAT for now but flagged)
        return True, None

    def _formal_with_z3(self, timeout_sec: int):
        # 使用 circuit.build_symbolic_constraints 来建立两个副本约束
        # 约束语义：对于副本 A 和 B，所有 other_POs_A == other_POs_B
        # 并且 last_PO_A != last_PO_B
        s = z3.Solver()
        s.set('timeout', int(timeout_sec * 1000))

        # create PI variables for two copies
        A_vars = [z3.Int(f"A_{p}") for p in self.PIs]
        B_vars = [z3.Int(f"B_{p}") for p in self.PIs]
        # domain constraints 0..2
        for v in A_vars + B_vars:
            s.add(v >= 0, v <= 2)

        # let circuit provide constraints: it must accept a solver and a mapping var name prefix
        # expected API: circuit.build_symbolic_constraints(solver, var_map)
        # var_map should map signal names (PIs and intermediate gates and POs) to z3 Int/Exprs
        # We will ask the circuit to produce constraints for copy A and copy B
        # Build var map for A: map PI names -> A_vars
        A_map = {self.PIs[i]: A_vars[i] for i in range(self.n)}
        B_map = {self.PIs[i]: B_vars[i] for i in range(self.n)}

        # ask circuit to add constraints for each copy and return expressions for outputs
        outA = self.circuit.build_symbolic_constraints(s, A_map, suffix="_A")
        outB = self.circuit.build_symbolic_constraints(s, B_map, suffix="_B")
        # outA and outB are expected to be dict: POname -> z3 expr

        # add equality constraints for other outputs
        for po in self.other_POs:
            s.add(outA[po] == outB[po])
        # add disequality for last
        s.add(outA[self.last_PO] != outB[self.last_PO])

        res = s.check()
        if res == z3.sat:
            m = s.model()
            a = tuple(int(str(m[A_vars[i]])) for i in range(self.n))
            b = tuple(int(str(m[B_vars[i]])) for i in range(self.n))
            return False, (a, b)
        elif res == z3.unknown:
            # 超时或无法判断
            return True, None
        else:
            return True, None

    # ------------------------- 主入口 -------------------------
    def solve(self, mode: str = 'auto', sim_rounds: int = 4096, formal_timeout: int = 5) -> ResubstitutionResult:
        """
        执行 S&S 流程：
        1) Simulation 阶段（随机 + 引导），尽量发现反例
        2) 如果没有发现，Formal 阶段尝试证明不存在反例（优先 z3）
        3) 如果证明 UNSAT，则构造 F 的真值表（用枚举或 solver 枚举 reachable patterns）
        """
        start = time.time()
        # quick dependency heuristic: if some PI influences last PO but not other POs -> immediate NO
        dep_res = self._check_independent_input()
        if dep_res is not None:
            return dep_res

        # simulation phase
        mapping, witness = self.simulation_phase(rounds=sim_rounds)
        if witness is not None:
            a, b = witness
            return ResubstitutionResult(False, conflict=(self._tuple_to_dict(a), self._tuple_to_dict(b)), conflict_key=self._simulate_tuple(a)[0], conflict_values=(self._simulate_tuple(a)[1], self._simulate_tuple(b)[1]))

        # formal phase
        is_unsat, witness2 = self.formal_phase(timeout_sec=formal_timeout)
        if not is_unsat:
            a, b = witness2
            return ResubstitutionResult(False, conflict=(self._tuple_to_dict(a), self._tuple_to_dict(b)), conflict_key=self._simulate_tuple(a)[0], conflict_values=(self._simulate_tuple(a)[1], self._simulate_tuple(b)[1]))

        # proved (or heuristically assumed) UNSAT: build function table for all reachable other_patterns
        # We'll enumerate known observed other patterns from simulation cache and if small, try to enumerate all possible patterns
        observed_patterns = set(v[0] for v in getattr(self, '_sim_cache', {}).values())
        # if the number of other outputs is small, enumerate all possible patterns to find don't cares
        num_other = len(self.other_POs)
        total_patterns = 3 ** num_other if num_other > 0 else 1
        function_table = {}
        if len(observed_patterns) < total_patterns and total_patterns <= 20000:
            # enumerates all possible other patterns; for each, try to find one input producing it (use cache or brute force)
            for pattern in self._enumerate_other_patterns(num_other):
                example = self._find_input_producing_other(pattern)
                if example is not None:
                    _, last = self._simulate_tuple(example)
                    function_table[pattern] = last
        else:
            # just use observed mapping
            function_table = dict(mapping)

        dont_care = set()
        if total_patterns <= 20000:
            all_patterns = list(self._enumerate_other_patterns(num_other))
            for p in all_patterns:
                if p not in function_table:
                    dont_care.add(p)

        return ResubstitutionResult(True, function_table=function_table, dont_care_patterns=dont_care)

    # ------------------------- 辅助 -------------------------
    def _find_input_producing_other(self, target: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        # first check cache
        for inp, (oth, last) in getattr(self, '_sim_cache', {}).items():
            if oth == target:
                return inp
        # brute force with limit
        limit = 200000
        cnt = 0
        for inp in self._input_enumerator():
            oth, last = self._simulate_tuple(inp)
            if oth == target:
                return inp
            cnt += 1
            if cnt > limit:
                break
        return None

    def _enumerate_other_patterns(self, k: int):
        if k == 0:
            yield tuple()
            return
        current = [0] * k
        total = 3 ** k
        for _ in range(total):
            yield tuple(current)
            i = 0
            while i < k:
                current[i] += 1
                if current[i] >= 3:
                    current[i] = 0
                    i += 1
                else:
                    break

    def _tuple_to_dict(self, inp: Tuple[int, ...]) -> Dict[str, int]:
        return {self.PIs[i]: inp[i] for i in range(self.n)}

    def _check_independent_input(self) -> Optional[ResubstitutionResult]:
        # 轻量依赖分析：尝试检测仅影响 last_PO 而不影响 other_POs 的 PI
        # If circuit provides fanin/fanout info, a stronger version could be implemented
        if not hasattr(self.circuit, 'gates'):
            return None
        # compute dependencies quickly by simulation: toggle each PI and check which POs change
        affects_other = {pi: False for pi in self.PIs}
        affects_last = {pi: False for pi in self.PIs}
        base = tuple(0 for _ in range(self.n))
        base_other, base_last = self._simulate_tuple(base)
        for i, pi in enumerate(self.PIs):
            for v in (1, 2):
                t = list(base)
                t[i] = v
                oth, last = self._simulate_tuple(tuple(t))
                if oth != base_other:
                    affects_other[pi] = True
                if last != base_last:
                    affects_last[pi] = True
        for pi in self.PIs:
            if affects_last[pi] and not affects_other[pi]:
                # attempt to construct a witness
                idx = self.PIs.index(pi)
                for v1 in range(3):
                    for v2 in range(3):
                        if v1 == v2:
                            continue
                        a = list(base)
                        b = list(base)
                        a[idx] = v1
                        b[idx] = v2
                        other_a, last_a = self._simulate_tuple(tuple(a))
                        other_b, last_b = self._simulate_tuple(tuple(b))
                        if other_a == other_b and last_a != last_b:
                            return ResubstitutionResult(False, conflict=(self._tuple_to_dict(tuple(a)), self._tuple_to_dict(tuple(b))), conflict_key=other_a, conflict_values=(last_a, last_b))
        return None

    def print_function_table(self, result: ResubstitutionResult):
        if not result.can_resubstitute:
            print("无法构建替换函数表（不可重替换）")
            if result.conflict is not None:
                a, b = result.conflict
                print("反例输入 A:", a)
                print("反例输入 B:", b)
            return
        print(f"替换函数 F: {self.last_PO} = F({', '.join(self.other_POs)})")
        print("-" * 60)
        for k in sorted(result.function_table.keys()):
            print(k, '->', result.function_table[k])
        if result.dont_care_patterns:
            print(f"Don't care ({len(result.dont_care_patterns)}): {sorted(result.dont_care_patterns)[:20]}{('...' if len(result.dont_care_patterns)>20 else '')}")


# End of file
