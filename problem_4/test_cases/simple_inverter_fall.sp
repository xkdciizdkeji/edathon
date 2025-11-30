
* Simple Inverter
.subckt INV OUT IN VDD GND
.inputs IN
.outputs OUT

MN1 OUT IN GND GND nmos W=1u L=0.1u
MP1 OUT IN VDD VDD pmos W=2u L=0.1u

.ends INV
