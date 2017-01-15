# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint("""
Example3:

Simple nonlinear system

mass on a spring with cubic characteristic curve
""")


# parameters

m = 1.0
c = 10.0

u1, fb = inputs('u1, fb') # external force and feedback

SUM = Blockfnc(u1 - c*fb)
DI = TFBlock((1/m)/(s**2), SUM.Y) # double integrator x_ddot -> x
CUB = Blockfnc(DI.Y**3) # nonlinear characteristic curve of the spring
loop(CUB.Y, fb)

u1fnc = stepfnc(2, 1)

t, states = blocksimulation(10, (u1, u1fnc)) # simulate 10 seconds

bo = compute_block_ouptputs(states)

# an undamped nonlinear oscillation takes place:

if __name__ == "__main__":
    pl.plot(t, bo[DI])
    pl.grid()
    pl.show()

