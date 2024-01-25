# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint(
    """
Example1:

step response of a first order linear transfer function (PT1)
"""
)


(u1,) = inputs("u1,")

PT1 = TFBlock(1 / (3 * s + 1), u1)  # gain: 1, time constant: 3

u1fnc = stepfnc(0.5, 1)

t, states = blocksimulation(10, (u1, u1fnc))  # integrate 10 seconds

bo = compute_block_ouptputs(states)

if __name__ == "__main__":
    pl.plot(t, bo[PT1])
    pl.grid()
    pl.show()
