# -*- coding: utf-8 -*-

from pyblocksim import *

import symbtools as st


print(
    """

Simple nonlinear system

mass on a spring with external force which comes from PID controller
"""
)


# parameters

m = 1.0
c = 10.0

w1, u1, fb = inputs("w1, u1, fb")  # external force and feedback

SUM = Blockfnc(u1 - c * fb)
DI = TFBlock((1 / m) / (s**2), SUM.Y)  # double integrator x_ddot -> x
loop(DI.Y, fb)

# displacement

x = DI.Y

e = w1 - x


# PID-controller
kp = 50
ki = 80.0
kd = 8.1

p_part = Blockfnc(kp * e)
i_part = TFBlock(ki / (s), e)
from ipHelp import IPS

# IPS()

d_part = Blockfnc(-kd * DI.denomBlock.requestDeriv(1))


controller_out = Blockfnc(p_part.Y + i_part.Y + d_part.Y)


loop(controller_out.Y, u1)


T0 = 0.5
T1 = T0 + 0.3
w1fnc = stepfnc(T0, 1)
w1fnc_v = np.vectorize(w1fnc)

tt, states = blocksimulation(3, (w1, w1fnc))  # simulate

bo = compute_block_outputs(states)


pl.rc("text", usetex=True)
pl.rcParams["font.size"] = 16
pl.rcParams["legend.fontsize"] = 16


pl.rcParams["figure.subplot.bottom"] = 0.10
pl.rcParams["figure.subplot.left"] = 0.10
pl.rcParams["figure.subplot.top"] = 0.98
pl.rcParams["figure.subplot.right"] = 0.98
pl.rcParams["font.family"] = "serif"

mm = 1.0 / 25.4  # mm to inch
scale = 1
fs = [100 * mm * scale, 60 * mm * scale]


color_list = ["#5778d5", "#2aa42a", "#000000", "#d45656"]

if 0:
    fig1 = pl.figure(1, figsize=fs)
    pl.plot(tt, w1fnc_v(tt), color=color_list[1], lw=1)
    pl.plot(tt, bo[DI], color=color_list[0], lw=2)
    pl.grid()
    pl.xticks([0, 1, 2, 3])
    pl.yticks([0, 0.5, 1])

    pl.axis([0, 2, -0.1, 1.3])

    pl.savefig("pid_step.pdf")


if 0:
    fs = [140 * mm * scale, 60 * mm * scale]
    fig1 = pl.figure(2, figsize=fs)

    pl.plot(tt, w1fnc_v(tt), color=color_list[1], lw=1)
    pl.plot(tt, bo[DI], color=color_list[0], lw=2)
    pl.grid()
    pl.xticks([0, 1, 2, 3])
    pl.yticks([0, 0.5, 1])
    pl.axis([0, 2, -0.1, 1.3])

    poly1 = st.trans_poly(t, 3, (T0, 0, 0, 0, 0), (T1, 1, 0, 0, 0))
    pw_poly = st.create_piecewise(t, (T0, T1), (0, poly1, 1))
    pw_poly_fnc = st.expr_to_func(t, pw_poly)

    pl.plot(tt, pw_poly_fnc(tt), color=color_list[3], lw=3)

    pl.savefig("pid_step2.pdf")

##


fs = [140 * mm * scale, 60 * mm * scale]
fig1 = pl.figure(3, figsize=fs)


pl.grid()
pl.xticks([0, 1, 2, 3])
pl.yticks([0, 0.5, 1])


poly1 = st.trans_poly(t, 3, (T0, 0, 0, 0, 0), (T1, 1, 0, 0, 0))
pw_poly = st.create_piecewise(t, (T0, T1), (0, poly1, 1))
pw_poly_fnc = st.expr_to_func(t, pw_poly)

poly1_fnc = st.expr_to_func(t, poly1)

pl.plot(tt, poly1_fnc(tt), "--", color=color_list[3], lw=1)
pl.plot(tt, pw_poly_fnc(tt), color=color_list[3], lw=3)


pl.axis([0, 2, -0.1, 1.3])

pl.savefig("pid_step3.pdf")


pl.show()
