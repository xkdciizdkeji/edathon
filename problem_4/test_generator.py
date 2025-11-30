"""
测试用例生成器 (Test Case Generator)

该模块用于生成各种复杂度的测试电路，包括：
1. 简单反相器 (Inverter)
2. 两级反相器 (Two-stage Inverter)
3. NAND门
4. NOR门
5. 复杂组合逻辑电路
6. 多级缓冲器链

每个测试用例包含:
- SPICE网表
- 输入跳变信息 (从0到1或从1到0)
- 预期的dominant devices (用于验证算法正确性)
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


@dataclass
class TestCase:
    """测试用例数据结构"""
    name: str                           # 测试用例名称
    description: str                    # 描述
    netlist: str                        # SPICE网表
    input_transitions: Dict[str, str]   # 输入跳变 {输入名: "rise"/"fall"}
    expected_dominant: List[str]        # 预期的dominant devices
    difficulty: str                     # 难度级别


def generate_inverter() -> TestCase:
    """
    生成简单反相器测试用例
    
    电路结构:
        IN --[MN1, MP1]--> OUT
        
    当IN从0→1时：
        - MN1导通(漏极电流大)，拉低OUT
        - MP1截止(漏极电流小)
        - dominant: MN1
        
    当IN从1→0时：
        - MP1导通(漏极电流大)，拉高OUT  
        - MN1截止(漏极电流小)
        - dominant: MP1
    """
    netlist = """
* Simple Inverter
.subckt INV OUT IN VDD GND
.inputs IN
.outputs OUT

MN1 OUT IN GND GND nmos W=1u L=0.1u
MP1 OUT IN VDD VDD pmos W=2u L=0.1u

.ends INV
"""
    
    return TestCase(
        name="simple_inverter_rise",
        description="简单反相器，输入上升沿(0→1)",
        netlist=netlist,
        input_transitions={"IN": "rise"},
        expected_dominant=["MN1"],
        difficulty="easy"
    )


def generate_inverter_fall() -> TestCase:
    """简单反相器，输入下降沿"""
    netlist = """
* Simple Inverter
.subckt INV OUT IN VDD GND
.inputs IN
.outputs OUT

MN1 OUT IN GND GND nmos W=1u L=0.1u
MP1 OUT IN VDD VDD pmos W=2u L=0.1u

.ends INV
"""
    
    return TestCase(
        name="simple_inverter_fall",
        description="简单反相器，输入下降沿(1→0)",
        netlist=netlist,
        input_transitions={"IN": "fall"},
        expected_dominant=["MP1"],
        difficulty="easy"
    )


def generate_two_stage_inverter() -> TestCase:
    """
    生成两级反相器测试用例
    
    电路结构:
        IN --[MN1, MP1]--> MID --[MN2, MP2]--> OUT
    
    当IN从0→1时：
        - 第一级: MN1导通拉低MID，MP1截止
        - 第二级: MID从1→0，所以MP2导通拉高OUT，MN2截止
        - dominant: MN1(拉低MID), MP2(拉高OUT)
    """
    netlist = """
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
    
    return TestCase(
        name="two_stage_inverter_rise",
        description="两级反相器，输入上升沿(0→1)，预期MN1和MP2为dominant",
        netlist=netlist,
        input_transitions={"IN": "rise"},
        expected_dominant=["MN1", "MP2"],
        difficulty="easy"
    )


def generate_two_stage_inverter_fall() -> TestCase:
    """两级反相器，输入下降沿"""
    netlist = """
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
    
    return TestCase(
        name="two_stage_inverter_fall",
        description="两级反相器，输入下降沿(1→0)，预期MP1和MN2为dominant",
        netlist=netlist,
        input_transitions={"IN": "fall"},
        expected_dominant=["MP1", "MN2"],
        difficulty="easy"
    )


def generate_nand2() -> TestCase:
    """
    生成2输入NAND门测试用例
    
    电路结构 (CMOS NAND2):
        VDD -- MP1(gate=A) -- OUT
            -- MP2(gate=B) -- OUT
        OUT -- MN2(gate=B) -- N1
        N1  -- MN1(gate=A) -- GND
    
    当A和B都从0→1时(输出从1→0):
        - MN1和MN2都导通，形成放电路径
        - MP1和MP2都截止
        - dominant: MN1, MN2 (串联放电)
    """
    netlist = """
* 2-input NAND gate
.subckt NAND2 OUT A B VDD GND
.inputs A B
.outputs OUT

* PMOS pull-up network (parallel)
MP1 OUT A VDD VDD pmos W=2u L=0.1u
MP2 OUT B VDD VDD pmos W=2u L=0.1u

* NMOS pull-down network (series)
MN2 OUT B N1 GND nmos W=2u L=0.1u
MN1 N1 A GND GND nmos W=2u L=0.1u

.ends NAND2
"""
    
    return TestCase(
        name="nand2_both_rise",
        description="2输入NAND门，A和B都从0→1(输出1→0)，预期MN1和MN2为dominant",
        netlist=netlist,
        input_transitions={"A": "rise", "B": "rise"},
        expected_dominant=["MN1", "MN2"],
        difficulty="medium"
    )


def generate_nand2_a_fall() -> TestCase:
    """NAND2门，只有A从1→0 (B保持1)"""
    netlist = """
* 2-input NAND gate
.subckt NAND2 OUT A B VDD GND
.inputs A B
.outputs OUT

* PMOS pull-up network (parallel)
MP1 OUT A VDD VDD pmos W=2u L=0.1u
MP2 OUT B VDD VDD pmos W=2u L=0.1u

* NMOS pull-down network (series)
MN2 OUT B N1 GND nmos W=2u L=0.1u
MN1 N1 A GND GND nmos W=2u L=0.1u

.ends NAND2
"""
    
    return TestCase(
        name="nand2_a_fall",
        description="2输入NAND门，A从1→0(B保持1)，输出0→1，预期MP1为dominant",
        netlist=netlist,
        input_transitions={"A": "fall"},
        expected_dominant=["MP1"],
        difficulty="medium"
    )


def generate_nor2() -> TestCase:
    """
    生成2输入NOR门测试用例
    
    电路结构 (CMOS NOR2):
        VDD -- MP1(gate=A) -- N1
        N1  -- MP2(gate=B) -- OUT
        OUT -- MN1(gate=A) -- GND
            -- MN2(gate=B) -- GND
    
    当A和B都从1→0时(输出从0→1):
        - MP1和MP2都导通，形成充电路径
        - MN1和MN2都截止
        - dominant: MP1, MP2 (串联充电)
    """
    netlist = """
* 2-input NOR gate
.subckt NOR2 OUT A B VDD GND
.inputs A B
.outputs OUT

* PMOS pull-up network (series)
MP1 N1 A VDD VDD pmos W=4u L=0.1u
MP2 OUT B N1 VDD pmos W=4u L=0.1u

* NMOS pull-down network (parallel)
MN1 OUT A GND GND nmos W=1u L=0.1u
MN2 OUT B GND GND nmos W=1u L=0.1u

.ends NOR2
"""
    
    return TestCase(
        name="nor2_both_fall",
        description="2输入NOR门，A和B都从1→0(输出0→1)，预期MP1和MP2为dominant",
        netlist=netlist,
        input_transitions={"A": "fall", "B": "fall"},
        expected_dominant=["MP1", "MP2"],
        difficulty="medium"
    )


def generate_buffer_chain() -> TestCase:
    """
    生成4级缓冲器链测试用例
    
    电路结构:
        IN --[INV1]--> N1 --[INV2]--> N2 --[INV3]--> N3 --[INV4]--> OUT
    
    当IN从0→1时：
        - INV1: MN1导通(N1从1→0)
        - INV2: MP2导通(N2从0→1)
        - INV3: MN3导通(N3从1→0)
        - INV4: MP4导通(OUT从0→1)
        - dominant: MN1, MP2, MN3, MP4
    """
    netlist = """
* 4-stage Buffer Chain
.subckt BUF4 OUT IN VDD GND
.inputs IN
.outputs OUT

* Stage 1 (INV)
MN1 N1 IN GND GND nmos W=1u L=0.1u
MP1 N1 IN VDD VDD pmos W=2u L=0.1u

* Stage 2 (INV)
MN2 N2 N1 GND GND nmos W=1u L=0.1u
MP2 N2 N1 VDD VDD pmos W=2u L=0.1u

* Stage 3 (INV)
MN3 N3 N2 GND GND nmos W=1u L=0.1u
MP3 N3 N2 VDD VDD pmos W=2u L=0.1u

* Stage 4 (INV)
MN4 OUT N3 GND GND nmos W=1u L=0.1u
MP4 OUT N3 VDD VDD pmos W=2u L=0.1u

.ends BUF4
"""
    
    return TestCase(
        name="buffer_chain_4stage",
        description="4级缓冲器链，输入上升沿，预期交替的MN/MP为dominant",
        netlist=netlist,
        input_transitions={"IN": "rise"},
        expected_dominant=["MN1", "MP2", "MN3", "MP4"],
        difficulty="medium"
    )


def generate_complex_aoi21() -> TestCase:
    """
    生成AOI21 (AND-OR-INVERT) 复合门测试用例
    
    逻辑功能: OUT = ~((A & B) | C)
    
    电路结构:
        VDD -- MP3(gate=C) -- N2
        N2  -- MP1(gate=A) -- OUT
            -- MP2(gate=B) -- OUT
        OUT -- MN1(gate=A) -- N1
        N1  -- MN2(gate=B) -- GND
        OUT -- MN3(gate=C) -- GND
    """
    netlist = """
* AOI21 gate: OUT = ~((A & B) | C)
.subckt AOI21 OUT A B C VDD GND
.inputs A B C
.outputs OUT

* PMOS network
MP3 N2 C VDD VDD pmos W=2u L=0.1u
MP1 OUT A N2 VDD pmos W=2u L=0.1u
MP2 OUT B N2 VDD pmos W=2u L=0.1u

* NMOS network  
MN1 OUT A N1 GND nmos W=1u L=0.1u
MN2 N1 B GND GND nmos W=1u L=0.1u
MN3 OUT C GND GND nmos W=2u L=0.1u

.ends AOI21
"""
    
    return TestCase(
        name="aoi21_ab_rise",
        description="AOI21门，A和B从0→1(C=0)，输出1→0，预期MN1和MN2为dominant",
        netlist=netlist,
        input_transitions={"A": "rise", "B": "rise"},
        expected_dominant=["MN1", "MN2"],
        difficulty="hard"
    )


def generate_complex_oai22() -> TestCase:
    """
    生成OAI22 (OR-AND-INVERT) 复合门测试用例
    
    逻辑功能: OUT = ~((A | B) & (C | D))
    """
    netlist = """
* OAI22 gate: OUT = ~((A | B) & (C | D))
.subckt OAI22 OUT A B C D VDD GND
.inputs A B C D
.outputs OUT

* PMOS network (series-parallel)
MP1 N1 A VDD VDD pmos W=4u L=0.1u
MP2 N1 B VDD VDD pmos W=4u L=0.1u
MP3 OUT C N1 VDD pmos W=4u L=0.1u
MP4 OUT D N1 VDD pmos W=4u L=0.1u

* NMOS network (parallel-series)
MN1 OUT A N2 GND nmos W=1u L=0.1u
MN2 OUT B N2 GND nmos W=1u L=0.1u
MN3 N2 C GND GND nmos W=1u L=0.1u
MN4 N2 D GND GND nmos W=1u L=0.1u

.ends OAI22
"""
    
    return TestCase(
        name="oai22_ac_rise",
        description="OAI22门，A和C从0→1(B=D=0)，输出1→0，预期MN1和MN3为dominant",
        netlist=netlist,
        input_transitions={"A": "rise", "C": "rise"},
        expected_dominant=["MN1", "MN3"],
        difficulty="hard"
    )


def generate_transmission_gate() -> TestCase:
    """
    生成传输门测试用例
    
    传输门由一个NMOS和一个PMOS并联组成，
    由互补的控制信号(EN和ENB)控制。
    """
    netlist = """
* Transmission Gate with Inverter Buffer
.subckt TGATE OUT IN EN VDD GND
.inputs IN EN
.outputs OUT

* Control signal inverter
MN_INV ENB EN GND GND nmos W=1u L=0.1u
MP_INV ENB EN VDD VDD pmos W=2u L=0.1u

* Transmission gate
MN_TG OUT IN GND GND nmos W=2u L=0.1u
MP_TG OUT IN VDD VDD pmos W=2u L=0.1u

* Output buffer
MN_BUF OUTB OUT GND GND nmos W=1u L=0.1u
MP_BUF OUTB OUT VDD VDD pmos W=2u L=0.1u

.ends TGATE
"""
    
    return TestCase(
        name="transmission_gate",
        description="传输门，输入上升沿，EN=1",
        netlist=netlist,
        input_transitions={"IN": "rise"},
        expected_dominant=["MN_TG", "MP_TG", "MN_BUF"],
        difficulty="hard"
    )


def generate_mux2() -> TestCase:
    """
    生成2选1多路选择器测试用例
    
    MUX2: OUT = S ? B : A
    使用传输门实现
    """
    netlist = """
* 2:1 MUX using transmission gates
.subckt MUX2 OUT A B S VDD GND
.inputs A B S
.outputs OUT

* S inverter
MN_SI SB S GND GND nmos W=1u L=0.1u
MP_SI SB S VDD VDD pmos W=2u L=0.1u

* TG for A (selected when S=0)
MN_A N1 A GND GND nmos W=1u L=0.1u
MP_A N1 A VDD VDD pmos W=2u L=0.1u

* TG for B (selected when S=1)
MN_B N1 B GND GND nmos W=1u L=0.1u
MP_B N1 B VDD VDD pmos W=2u L=0.1u

* Output inverter
MN_OUT OUT N1 GND GND nmos W=1u L=0.1u
MP_OUT OUT N1 VDD VDD pmos W=2u L=0.1u

.ends MUX2
"""
    
    return TestCase(
        name="mux2_a_rise",
        description="2选1MUX，A输入上升沿(S=0选择A)",
        netlist=netlist,
        input_transitions={"A": "rise"},
        expected_dominant=["MN_A", "MP_A", "MP_OUT"],
        difficulty="hard"
    )


def generate_large_circuit() -> TestCase:
    """
    生成一个较大的组合逻辑电路测试用例
    包含多个逻辑门的级联
    """
    netlist = """
* Large combinational circuit
* Logic: OUT = ~(((A & B) | C) & (D | E))
.subckt COMPLEX OUT A B C D E VDD GND
.inputs A B C D E
.outputs OUT

* Stage 1: NAND2 for (A & B) -> N1 = ~(A & B)
MP1_1 N1 A VDD VDD pmos W=2u L=0.1u
MP1_2 N1 B VDD VDD pmos W=2u L=0.1u
MN1_1 N1 A T1 GND nmos W=2u L=0.1u
MN1_2 T1 B GND GND nmos W=2u L=0.1u

* Stage 2: NOR2 for ~(~(A&B) & ~C) = (A&B)|C -> N2
MP2_1 T2 N1 VDD VDD pmos W=4u L=0.1u
MP2_2 N2 C T2 VDD pmos W=4u L=0.1u
MN2_1 N2 N1 GND GND nmos W=1u L=0.1u
MN2_2 N2 C GND GND nmos W=1u L=0.1u

* Stage 3: NAND2 for (D | E) using De Morgan -> N3 = ~(~D & ~E) = D|E via NOR+INV
* First NOR: ~(D|E) = ~D & ~E
MP3_1 T3 D VDD VDD pmos W=4u L=0.1u
MP3_2 N3_T D T3 VDD pmos W=4u L=0.1u
MN3_1 N3_T D GND GND nmos W=1u L=0.1u
MN3_2 N3_T E GND GND nmos W=1u L=0.1u

* Inverter for D|E -> N3
MN3_3 N3 N3_T GND GND nmos W=1u L=0.1u
MP3_3 N3 N3_T VDD VDD pmos W=2u L=0.1u

* Stage 4: NAND for final output: ~(N2 & N3)
MP4_1 OUT N2 VDD VDD pmos W=2u L=0.1u
MP4_2 OUT N3 VDD VDD pmos W=2u L=0.1u
MN4_1 OUT N2 T4 GND nmos W=2u L=0.1u
MN4_2 T4 N3 GND GND nmos W=2u L=0.1u

.ends COMPLEX
"""
    
    return TestCase(
        name="large_combinational",
        description="大型组合逻辑电路，A和B从0→1",
        netlist=netlist,
        input_transitions={"A": "rise", "B": "rise"},
        expected_dominant=["MN1_1", "MN1_2", "MN2_1", "MP4_1"],
        difficulty="expert"
    )


def generate_all_test_cases() -> List[TestCase]:
    """生成所有测试用例"""
    return [
        generate_inverter(),
        generate_inverter_fall(),
        generate_two_stage_inverter(),
        generate_two_stage_inverter_fall(),
        generate_nand2(),
        generate_nand2_a_fall(),
        generate_nor2(),
        generate_buffer_chain(),
        generate_complex_aoi21(),
        generate_complex_oai22(),
        generate_transmission_gate(),
        generate_mux2(),
        generate_large_circuit(),
    ]


def save_test_cases(test_cases: List[TestCase], output_dir: str):
    """
    保存测试用例到文件
    
    参数:
        test_cases: 测试用例列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个测试用例的网表
    for tc in test_cases:
        netlist_file = os.path.join(output_dir, f"{tc.name}.sp")
        with open(netlist_file, 'w', encoding='utf-8') as f:
            f.write(tc.netlist)
        
        # 保存测试用例元信息
        info_file = os.path.join(output_dir, f"{tc.name}_info.json")
        info = {
            "name": tc.name,
            "description": tc.description,
            "input_transitions": tc.input_transitions,
            "expected_dominant": tc.expected_dominant,
            "difficulty": tc.difficulty
        }
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    # 保存测试用例汇总
    summary_file = os.path.join(output_dir, "summary.json")
    summary = {
        "total_cases": len(test_cases),
        "cases": [
            {
                "name": tc.name,
                "description": tc.description,
                "difficulty": tc.difficulty,
                "expected_dominant_count": len(tc.expected_dominant)
            }
            for tc in test_cases
        ]
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"已保存 {len(test_cases)} 个测试用例到 {output_dir}")


if __name__ == "__main__":
    # 生成并保存所有测试用例
    test_cases = generate_all_test_cases()
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_cases")
    
    save_test_cases(test_cases, output_dir)
    
    # 打印测试用例摘要
    print("\n" + "="*60)
    print("测试用例摘要")
    print("="*60)
    for tc in test_cases:
        print(f"\n[{tc.difficulty.upper()}] {tc.name}")
        print(f"  描述: {tc.description}")
        print(f"  输入跳变: {tc.input_transitions}")
        print(f"  预期dominant: {tc.expected_dominant}")
