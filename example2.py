# -*- coding: utf-8 -*-

from pyblocksim import *

mainprint("""
Example2 : linear system consiting of various blocks

 (coupled linear transfer functions)
""")

fb1, fb2, w, z11, z21, z12, z22 = inputs('fb1, fb2, w, z11, z21, z12, z22')

# The two PI controllers (naively parameterized by neglecting the coupling):
KR1 = 1.7
TN1 = 1.29

KR2 = 0.57
TN2 = 1.29

### Plant
T1 = 1.0

# equilibrium pojnt 1
K1, K2, K3, K4 = 0.4, 1.2, -0.8, -0.2

# equilibrium point 2
#K1, K2, K3, K4 = 0.4, 1.2, -1.28, -0.32

#switch of coupling:
#K3, K4 = 0,0


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

thestep = stepfnc(1.0, 1)

t, states = blocksimulation(40, (w, thestep), dt=.05)

bo = compute_block_ouptputs(states)

if __name__ == "__main__":
    pl.plot(t, bo[SUM1])
    pl.plot(t, bo[P22])
    pl.plot(t, bo[P11])
    pl.grid()
    pl.show()

