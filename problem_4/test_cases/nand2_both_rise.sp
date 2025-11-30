
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
