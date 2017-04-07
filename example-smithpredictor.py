# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint("""
Example1:

step response of a first order linear transfer function (PT1)
""")


u1, fb0, fb1, fb2, fb3 = inputs('u1, fb0, fb1, fb2, fb3')


K = 0.3
T = 3
Kp = 10/K

# delaytime
DT = 0.5


diff1 = Blockfnc(u1 - fb0)
diff2 = Blockfnc(u1 - fb1)

diff3a = Blockfnc(u1 - fb2)
diff3b = Blockfnc(diff3a.Y - fb3)

# PI controller
PIc1 = TFBlock(Kp*(1 + 1/(T*s)), diff1.Y)
PIc2 = TFBlock(Kp*(1 + 1/(T*s)), diff2.Y)
PIc3 = TFBlock(Kp*(1 + 1/(T*s)), diff3b.Y)

# Plant 1 (undelayed, controlled -> as desired)
PT1a = TFBlock(K/(s*T + 1), PIc1.Y)  # gain: 0.3, time constant: 3
loop(PT1a.Y, fb0)


# Plant 2 (delayed, controlled -> unstable)
Delay1 = DelayBlock(DT, PIc2.Y)
PT1b = TFBlock(K/(s*T + 1), Delay1.Y)  # gain: 0.3, time constant: 3
loop(PT1b.Y, fb1)


# Plant 3 (delayed controlled with smith-predictor)

# two blocks: plant and prediction modell

PT1c_plant = TFBlock(Kp*(1 + 1/(T*s)), PIc3.Y)
PT1c_pred = TFBlock(Kp*(1 + 1/(T*s)), PIc3.Y)



# two delay-blocks (one for the plant, another for the predictor)
Delay2a = DelayBlock(DT, PT1c_plant.Y)
Delay2b = DelayBlock(DT, PT1c_pred.Y)
#
#diff_sp1 = Blockfnc(PT1c_pred.Y - Delay2b.Y)

# outer loop:
# loop(Delay2a.Y, fb2)

# inner loop
# loop(diff_sp1.Y, fb3)


# Reference uncontrolled, delayed system

Delay3 = DelayBlock(DT, u1)
PT1d = TFBlock(K/(s*T + 1), Delay3.Y)



if 0:

    Delay1 = DelayBlock(2.5, u1)

    PT1b = TFBlock(1/(3*s + 1), Delay1.Y)  # gain: 1, time constant: 3

u1fnc = stepfnc(0.7, 1)

t, states = blocksimulation(10, (u1, u1fnc))  # integrate 10 seconds

bo = compute_block_ouptputs(states)

if __name__ == "__main__":
    pl.figure()
    pl.plot(t, bo[PT1a])
    pl.plot(t, bo[PT1d])
    pl.title("controlled vs. uncontrolled ")
    pl.figure()
    pl.plot(t, bo[PT1a], label="undelayed")
    pl.plot(t, bo[PT1b], label="delayed")
    # pl.plot(t, bo[Delay2a], label="with Smith predictor")
    pl.title("undelayed vs delayed")

    pl.grid()
    pl.show()
    from IPython import embed as IPS
    IPS()

