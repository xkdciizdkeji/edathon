
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
