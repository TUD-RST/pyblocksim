import unittest
import pyblocksim as pbs
import numpy as np
from matplotlib import pyplot as plt

from ipydex import IPS



class TestTD1(unittest.TestCase):
    def setUp(self):
        pbs.restart()

    def test_limit1(self):

        x = pbs.td.sp.Symbol("x")
        expr = pbs.td.limit(x, -3, 7, -1, 4)
        fnc = pbs.td.st.expr_to_func(x, expr)
        xx = np.linspace(-10, 10, 1000)

        if 0:
            plt.plot(xx, fnc(xx))
            plt.xticks(np.arange(-10, 10))
            plt.grid()
            plt.show()

        self.assertEqual(fnc(-4), -1)
        self.assertAlmostEqual(fnc(-2.8), -.9)
        self.assertAlmostEqual(fnc(0), 0.5)
        self.assertAlmostEqual(fnc(6.9), 3.95)
        self.assertEqual(fnc(7.1), 4)


    def test_block_construction1(self):

        block_class = pbs.td.new_TDBlock(2)
        dtPT1_1 = pbs.td.dtPT1(params=dict(K=1, T1=1))
        dtPT1_2 = pbs.td.dtPT1(params=dict(K=1, T1=1))

        self.assertEqual(pbs.td.dtPT1.instance_counter, 2)
        q = block_class()

        self.assertEqual(q.x2.name, "x4")

    def test_block_simulation1(self):
        u_amplitude = 10
        u_step_time = 1
        T1 = 1
        u1_expr = pbs.td.td_step(pbs.td.k, u_step_time, u_amplitude)
        dtPT1_1 = pbs.td.dtPT1(input1=u1_expr, params=dict(K=1, T1=T1))
        dtPT1_2 = pbs.td.dtPT1(input1=dtPT1_1.Y, params=dict(K=1, T1=T1))


        dtPT2_1 = pbs.td.dtPT2(input1=u1_expr, params=dict(K=1, T1=T1, T2=T1))
        # dtPT2_2 = pbs.td.dtPT2Euler(input1=u1_expr, params=dict(K=1, T1=T1, T2=T1))
        dtsigm_1 = pbs.td.dtSigmoid(input1=u1_expr, params=dict(K=1, T_trans=3, sens=.1))

        kk, xx, bo = pbs.td.blocksimulation(100)

        if 0:
            from matplotlib import pyplot as plt
            T = pbs.td.T
            if 0:
                plt.plot(kk*T, xx[:, [0, 1, 2]], marker=".")
                plt.plot(kk*T, xx[:,0]*0 + (1-np.exp(-1))*u_amplitude, "k-")
            if 1:
                plt.plot(kk*T, xx[:, 2], marker=".")
                plt.plot(kk*T, xx[:, 4:], marker=".")
            plt.grid()
            plt.show()

        eval_k = int(u_step_time + T1/pbs.td.T)

        # IPS()

        # evaluate 63% criterion
        self.assertAlmostEqual(xx[eval_k, 0], (1 - np.exp(-1))*u_amplitude)


    def test_block_DirectionSensitiveSigmoid(self):

        T = pbs.td.T
        k = pbs.td.k
        T_trans_pos = 3
        T_trans_neg = 5
        u_amplitude = 20

        step1 = 10
        step2 = T_trans_pos/T + step1 + 25

        u1_expr = pbs.sp.Piecewise((0, k < step1), (u_amplitude, k < step2), (0, True))

        dss_1 = pbs.td.dtDirectionSensitiveSigmoid(
            input1=u1_expr,
            params=dict(K=1, T_trans_pos=T_trans_pos, T_trans_neg=T_trans_neg, sens=.1, f_wait_neg=0.3)
        )

        kk, xx, bo = pbs.td.blocksimulation(int(step2 + T_trans_neg/T)+10)

        steps_start = np.r_[step1*T, step2*T]
        steps_end = steps_start + np.r_[T_trans_pos, T_trans_neg]

        if 0:
            plt.plot(kk*T, dss_1.output_res, marker=".")
            plt.plot(kk*T, xx[:, 4], marker=".")
            plt.vlines(steps_start, ymin=-1, ymax=u_amplitude, colors="tab:pink")
            plt.vlines(steps_end, ymin=-1, ymax=u_amplitude, colors="k")
            plt.grid()
            plt.show()
