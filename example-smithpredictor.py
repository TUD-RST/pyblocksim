# -*- coding: utf-8 -*-

from pyblocksim import *

from ipHelp import IPS

from pyblocksim.core import NILBlock

mainprint("""
Example1:

Stepresponse of a simple 1st order system with delay (with PI controller
and with Smith-predictor)
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
# PIc3 = TFBlock(Kp*(1 + 1/(T*s)), diff3a.Y)

# Plant 1 (undelayed, controlled -> as desired)
PT1a = TFBlock(K/(s*T + 1), PIc1.Y)  # gain: 0.3, time constant: 3
loop(PT1a.Y, fb0)


# Plant 2 (delayed, controlled -> unstable)

PT1b = TFBlock(K/(s*T + 1), PIc2.Y)  # gain: 0.3, time constant: 3
Delay1 = DelayBlock(DT, PT1b.Y)
loop(Delay1.Y, fb1)


if 1:

    # Plant 3 (delayed controlled with smith-predictor)

    # two PT1-blocks: plant and prediction modell
    # two delay-blocks (one for the plant, another for the predictor)

    PT1c_plant = TFBlock(K/(s*T + 1), PIc3.Y)
    Delay2a = DelayBlock(DT, PT1c_plant.Y)
    loop(Delay2a.Y, fb2)

    PT1c_pred = TFBlock(K/(s*T + 1), PIc3.Y)
    Delay2b = DelayBlock(DT, PT1c_pred.Y)

    # difference-block (within the Smith Predictor)
    diff_sp1 = Blockfnc(PT1c_pred.Y - Delay2b.Y)

# inner loop
loop(diff_sp1.Y, fb3)

# outer loop:
# loop(Delay2a.Y, fb2)

# Reference uncontrolled, delayed system

Delay3 = DelayBlock(DT, u1)
PT1d = TFBlock(K/(s*T + 1), Delay3.Y)

u1fnc = stepfnc(0.7, 1)

t, states = blocksimulation(4, (u1, u1fnc))  # integrate

bo = compute_block_ouptputs(states)

if __name__ == "__main__":
    pl.figure()
    pl.plot(t, bo[PT1a])
    pl.plot(t, bo[PT1d])
    pl.grid()
    pl.title("controlled vs. uncontrolled ")

    pl.figure()
    pl.plot(t, bo[PT1a], label="undelayed")
    pl.plot(t, bo[PT1b], label="delayed")
    pl.plot(t, bo[PT1c_plant], 'r:', label="with Smith predictor")
    pl.legend()
    pl.grid()
    pl.title("undelayed controlled vs delayed controlled")

    pl.show()
    from IPython import embed as IPS

