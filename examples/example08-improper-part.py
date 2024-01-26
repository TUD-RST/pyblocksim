# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint(
    """
Example08:

step response of loop with an improper (derivative link) part


(this does not yet work -> introduce a small time constant (t0))
"""
)


T0 = 1e-3

r1, z1, tmp1, tmp2 = inputs("r1, z1, tmp1, tmp2")

sum0 = Blockfnc(r1 + tmp2)

G1 = TFBlock((s - 2) / (1 + T0 * s), sum0.Y)  # approximate D-link  with DT1-link
sum1 = Blockfnc(G1.Y + z1 - tmp1)

G2 = TFBlock(2 / (3 * s), sum1.Y)
G3 = TFBlock(1 / (s + 1), G2.Y)


G4 = TFBlock((2 * s**2 - 2 * s - 4) / (s**2 + 5 * s + 6), r1)

loop(G2.Y, tmp2)
loop(G3.Y, tmp1)


u1fnc = stepfnc(0.7, 1)

t, states = blocksimulation(10, (r1, u1fnc))  # integrate 10 seconds

bo = compute_block_outputs(states)

if __name__ == "__main__":
    pl.plot(t, bo[G2])
    pl.plot(t, bo[G4])
    pl.grid()
    pl.show()
