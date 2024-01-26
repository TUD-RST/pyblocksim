# -*- coding: utf-8 -*-

from pyblocksim import *


mainprint(
    """
Example4:

Flatness based control of a linear plant

(another academic exercise)

"""
)


# parameters

k = 20  # controller gain

y_ref, u_traj, fb = inputs("y_ref, u_traj, fb")  # external force and feedback

SUM = Blockfnc((y_ref - fb) * k + u_traj)

# Plant
P_tf = 2 * (s + 1) / ((s**2 - 2 * s + 2) * (s + 5))
PLANT = TFBlock(P_tf, SUM.Y)  # double integrator x_ddot -> x

# close the loop
loop(PLANT.Y, fb)


# input signals
# the flat output is called q here
q_A = 0  # beginning
q_E = -16  # end
T = 1  # Transition Time

traj_expr1 = q_A + (q_E - q_A) * ((t / T) ** 3 * (10 - 15 * t / T + 6 * (t / T) ** 2))

traj_expr2 = sp.Piecewise((0, t < 0), (traj_expr1, t <= T), (q_E, True))

traj = Trajectory(traj_expr2, 3)

y_ref_fnc = traj.combined_trajectories(2 * (s + 1))  # the reference of PLANT.Y

# y_ref_fnc = stepfnc(0, -32)  # for comparism

u_traj_fnc = traj.combined_trajectories((s**2 - 2 * s + 2) * (s + 5))


t, states = blocksimulation(10, {u_traj: u_traj_fnc, y_ref: y_ref_fnc})  # simulate 10 seconds
t = t.flatten()

bo = compute_block_outputs(states)


if __name__ == "__main__":
    # desired and simulated values are quite close
    # (the combination of feed forward and feedback controller works good)

    pl.plot(t, bo[PLANT], label="simulation")
    pl.plot(t, [y_ref_fnc(ti) for ti in t], label="desired")
    pl.legend()
    pl.grid()
    # pl.plot(t, [u_traj_fnc(ti) for ti in t]) # the precalculated input
    pl.show()
