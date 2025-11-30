
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
