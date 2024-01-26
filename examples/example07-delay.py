# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint(
    """
Example1:

step response of a first order linear transfer function (PT1)
"""
)


(u1,) = inputs("u1,")

PT1a = TFBlock(1 / (3 * s + 1), u1)  # gain: 1, time constant: 3
Delay1 = DelayBlock(2.5, u1)

PT1b = TFBlock(1 / (3 * s + 1), Delay1.Y)  # gain: 1, time constant: 3

u1fnc = stepfnc(0.7, 1)

t, states = blocksimulation(10, (u1, u1fnc))  # integrate 10 seconds

bo = compute_block_outputs(states)

if __name__ == "__main__":
    pl.plot(t, bo[PT1a])
    pl.plot(t, bo[PT1b])
    pl.grid()
    pl.show()
