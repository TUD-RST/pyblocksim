# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint(
    """
Example1:

This example shows how to use the output derivative of a transfer function

"""
)

u1, u2 = inputs("u1, u2")

TF1 = TFBlock((s - 3) / ((3 * s + 1) * (s**2 + s + 4)), u1)

Ydot = TF1.get_output_deriv(1)
Yddot = TF1.get_output_deriv(2)

meas1 = Blockfnc(1 * Ydot)

Integ1 = TFBlock(1 / s, Ydot)
Integ2 = TFBlock(1 / s, Yddot)

meas2 = Blockfnc(1 * Yddot)

square_impulse = Blockfnc(u1 - u2)

Double_Integ = TFBlock(1 / s**2, square_impulse.Y)  # Yddot)

u1fnc = stepfnc(0.5, 1)
u2fnc = stepfnc(0.55, 1)

t, states = blocksimulation(10, {u1: u1fnc, u2: u2fnc})  # integrate 10 seconds

bo = compute_block_outputs(states)

if __name__ == "__main__":
    pl.plot(t, bo[TF1], "b-", lw=4)
    pl.plot(t, bo[Integ1], "r-", lw=1.5)
    pl.plot(t, bo[meas1], "b-", lw=4)
    pl.plot(t, bo[Integ2], "r-", lw=1.5)
    pl.plot(t, bo[Double_Integ], "k-", lw=1)
    pl.plot(t, bo[meas2], "g-", lw=1)
    pl.title("blue and red curves should match")
    pl.grid()
    pl.show()
