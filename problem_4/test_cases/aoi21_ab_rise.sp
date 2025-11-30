
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
