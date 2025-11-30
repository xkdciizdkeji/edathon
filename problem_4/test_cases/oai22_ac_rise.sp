
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
