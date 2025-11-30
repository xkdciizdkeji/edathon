
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
