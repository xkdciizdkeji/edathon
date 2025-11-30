
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
