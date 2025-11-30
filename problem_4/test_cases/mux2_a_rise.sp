
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
