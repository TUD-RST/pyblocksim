# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint("""
Example2b : linear system consisting of various blocks

 (coupled linear transfer functions)

 Here we additionally extract the statepace model and simulate it
 with the control library (sampled with a zero order hold).
""")

fb1, fb2, w, z11, z21, z12, z22 = inputs('fb1, fb2, w, z11, z21, z12, z22')

# The two PI controllers (naively parameterized by neglecting the coupling):
KR1 = 1.7
TN1 = 1.29

KR2 = 0.57
TN2 = 1.29

# ## Plant
T1 = 1.0

# equilibrium pojnt 1
K1, K2, K3, K4 = 0.4, 1.2, -0.8, -0.2

# equilibrium point 2
# K1, K2, K3, K4 = 0.4, 1.2, -1.28, -0.32

# switch of coupling:
# K3, K4 = 0,0


DIF1 = Blockfnc(w - fb1)
DIF2 = Blockfnc(- fb2)

PI1 = TFBlock(KR1*(1+1/(s*TN1)), DIF1.Y)
PI2 = TFBlock(KR2*(1+1/(s*TN2)), DIF2.Y)

SUM11 = Blockfnc(PI1.Y + z11)
SUM21 = Blockfnc(PI1.Y + z21)

SUM12 = Blockfnc(PI2.Y + z12)
SUM22 = Blockfnc(PI2.Y + z22)

P11 = TFBlock( K1/s         , SUM11.Y)
P21 = TFBlock( K4/(1+s*T1)  , SUM21.Y)
P12 = TFBlock( K3/s         , SUM12.Y)
P22 = TFBlock( K2/s         , SUM22.Y)

SUM1 = Blockfnc(P11.Y + P12.Y)
SUM2 = Blockfnc(P22.Y + P21.Y)

loop(SUM1.Y, fb1)
loop(SUM2.Y, fb2)

# output of final summation blocks is system output
sys_output = sp.Matrix([SUM1.Y, SUM2.Y])

thestep = stepfnc(1.0, 1)


tt, states = blocksimulation(40, (w, thestep), dt=.05)

bo = compute_block_ouptputs(states)



# #### additional code for comparision

A, B, C, D = get_linear_ct_model(theStateAdmin, sys_output)

# create a control system with the python-control-toolbox, see
# https://github.com/python-control/python-control
import control
cs = control.StateSpace(A, B, C, D)

# use default method (zero order hold ("zoh"))
cs_dt = control.sample_system(cs, 0.05)


# calculate step-response for first input
__, yy_dt = control.step_response(cs_dt, T=tt, input=0)

# in the time-discrete case the result is shortened by one value. -> repeat the last one.
yy_dt = np.column_stack((yy_dt, yy_dt[:, -1]))


# __, yy_dt = control.step_response(cs,T=tt, input=0)


# #### end of additional code


if __name__ == "__main__":
    pl.plot(tt, bo[SUM1], "b-", lw=3, label="$y_1$ (pyblocksim)")
    pl.plot(tt, bo[SUM2], "r-", lw=3, label="$y_2$ (pyblocksim)")
    # +1 because the step_response starts the step at t=0
    pl.plot(tt+1, yy_dt[0, :], 'c--', label="$y_1$ (step_response)")
    pl.plot(tt+1, yy_dt[1, :], 'g--', label="$y_2$ (step_response)")
    pl.legend(loc="center right")
    pl.grid()
    pl.show()

