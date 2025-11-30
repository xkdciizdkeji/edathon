
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
