import unittest
import pyblocksim as pbs
import symbtools as st
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

from ipydex import Container, IPS, activate_ips_on_exception


class TestTD1(unittest.TestCase):
    def setUp(self):
        pbs.restart()

    def test_01__limit1(self):

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


    def test_02__block_construction1(self):

        block_class = pbs.td.new_TDBlock(2)
        dtPT1_1 = pbs.td.dtPT1(params=dict(K=1, T1=1))
        dtPT1_2 = pbs.td.dtPT1(params=dict(K=1, T1=1))

        self.assertEqual(pbs.td.dtPT1.instance_counter, 2)
        q = block_class()

        self.assertEqual(q.x2.name, "x4")

    def test_03a__block_simulation1(self):
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

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(100, rhs_options={"use_sp2c": True})
        self.assertAlmostEqual(xx2[eval_k, 0], (1 - np.exp(-1))*u_amplitude)

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_03b__block_simulation2(self):
        # now with static block
        u_amplitude = 10
        u_step_time = 1
        T1 = 1
        u1_expr = pbs.td.td_step(pbs.td.k, u_step_time, u_amplitude)
        dtPT1_1 = pbs.td.dtPT1(input1=u1_expr, params=dict(K=1, T1=T1))
        dtPT1_2 = pbs.td.dtPT1(input1=dtPT1_1.Y, params=dict(K=2, T1=T1))

        static_block = pbs.td.StaticBlock(output_expr=dtPT1_1.Y + dtPT1_2.Y**2)

        kk, xx, bo = pbs.td.blocksimulation(100)

        if 0:
            from matplotlib import pyplot as plt
            T = pbs.td.T
            if 1:
                plt.plot(kk*T, bo[dtPT1_1], marker=".")
                plt.plot(kk*T, bo[dtPT1_2], marker=".")
                plt.plot(kk*T, bo[static_block], marker=".")
            plt.grid()
            plt.show()

        # evaluate correct static calculation
        self.assertGreater(bo[static_block][-1], 409)
        self.assertLess(bo[static_block][-1], 410)

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(100, rhs_options={"use_sp2c": True})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_04a__modify_input_after_first_simulation(self):
        """
        Here we test a mode where the input function is evaluated separately.
        """
        T = pbs.td.T
        u_amplitude = 10
        u_step_time = 1*T
        T1 = 1
        u1_expr = pbs.td.td_step(pbs.td.k, u_step_time/T, u_amplitude)

        dtPT1_1 = pbs.td.dtPT1(input1=u1_expr, params=dict(K=1, T1=T1))
        dtPT1_2 = pbs.td.dtPT1(input1=dtPT1_1.Y, params=dict(K=2, T1=T1))

        static_block = pbs.td.StaticBlock(output_expr=dtPT1_1.Y + dtPT1_2.Y**2)

        N_steps = int(20/T)
        kk, xx, bo = pbs.td.blocksimulation(N_steps)

        if 0:
            from matplotlib import pyplot as plt
            T = pbs.td.T
            plt.plot(kk*T, bo[dtPT1_1], marker=".")
            plt.plot(kk*T, bo[dtPT1_2], marker=".")
            plt.plot(kk*T, bo[static_block], marker=".")
            plt.grid()
            plt.show()

        # evaluate correct static calculation
        self.assertGreater(bo[static_block][-1], 409)
        self.assertLess(bo[static_block][-1], 410)

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

        # now redefine the input
        u1_expr = pbs.td.td_step(pbs.td.k, u_step_time, u_amplitude) - 0.1*pbs.td.td_step(pbs.td.k, (u_step_time + 8)/T, u_amplitude)

        dtPT1_1.set_inputs(input1=u1_expr)
        pbs.td.generate_input_func()

        # now simulate again but reuse rhs
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        if 0:
            from matplotlib import pyplot as plt
            T = pbs.td.T
            plt.plot(kk*T, bo[dtPT1_1], marker=".")
            plt.plot(kk*T, bo[dtPT1_2], marker=".")
            plt.plot(kk*T, bo[static_block], marker=".")
            plt.grid()
            plt.show()

        # with the new input the output goes down after 8s (80 steps)
        self.assertGreater(bo[static_block][75], 405)
        self.assertLess(bo[static_block][175], 334)
