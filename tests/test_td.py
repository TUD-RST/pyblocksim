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

    def test_03__block_simulation1(self):
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

    def test_04__block_simulation2(self):
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

    def test_block_05a__DirectionSensitiveSigmoid(self):

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

        N_steps = int(step2 + T_trans_neg/T)+10
        kk, xx, bo = pbs.td.blocksimulation(N_steps)

        steps_start = np.r_[step1*T, step2*T]
        steps_end = steps_start + np.r_[T_trans_pos, T_trans_neg]

        if 0:
            plt.plot(kk*T, dss_1.output_res, marker=".")
            plt.plot(kk*T, xx[:, 4], marker=".")
            plt.vlines(steps_start, ymin=-1, ymax=u_amplitude, colors="tab:pink")
            plt.vlines(steps_end, ymin=-1, ymax=u_amplitude, colors="k")
            plt.grid()
            plt.show()

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_block_05b__Sufenta(self):

        dc = Container()

        dc.cr1 = dc.cr2 = 0.3
        T = pbs.td.T
        t = pbs.t

        T_end = 120
        tt = np.arange(0, int(T_end/T) + 1)*T

        u_expr_sufenta = sp.Piecewise((dc.cr1, apx(t, 5)), (dc.cr2, apx(t, 40)), (0, True))
        u_func = st.expr_to_func(t, u_expr_sufenta)

        params = dict(
            rise_time = 5,
            down_slope = -.8/15,
            active_time_coeff = 100,
            dose_gain = 0.5/0.3  # achieve output of 0.5 for 0.3 mg/kgKG
        )

        sufenta_block = pbs.td.dtSufenta(input1=u_expr_sufenta, params=params)

        N_steps = int(90/T)
        kk, xx, bo = pbs.td.blocksimulation(N_steps)

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_block_05c__Akrinor(self):

        T1 = 5 # dc.t_cr1
        T2 = 22.5 # dc.t_cr2

        # see notebook 07c_akrinor for where this comes from:
        acrinor_block_dose_gain = 5.530973451327434
        T = pbs.td.T
        t = pbs.t

        body_mass = 70

        relative_dose_akri = 0.02 # dc.cr1

        dose_akri = relative_dose_akri * body_mass

        u_expr_acrinor = sp.Piecewise((dose_akri, apx(t, T1)), (dose_akri, apx(t, T2)), (0, True))

        l1 = pbs.td.get_loop_symbol()
        bp_sum  = pbs.td.StaticBlock(output_expr=60 + l1)
        bp_delay_block = pbs.td.dtDelay1(input1=bp_sum.Y)

        T_end = 90

        params = dict(
            T_75=5,  # min
            T_plateau = 30,  # min (including rising phase)
            down_slope = -1,  # mmHg/min
            body_mass = 70,
            dose_gain = acrinor_block_dose_gain, # [1/(ml/kgKG)]
        )

        acrinor_block = pbs.td.dtAcrinor(input1=u_expr_acrinor, input2=bp_delay_block.Y, params=params)
        pbs.td.set_loop_symbol(l1, acrinor_block.Y)

        N_steps = int(T_end/T)

        # activate_ips_on_exception()

        # test for a bug that rhs returns different expressions each time
        test_expr1 = acrinor_block.rhs(0, (0,)*11)[2].args[0].args[1]
        test_expr2 = acrinor_block.rhs(0, (0,)*11)[2].args[0].args[1]

        self.assertEqual(test_expr1, test_expr2)

        # for the numeric comparison simulate with sympy_to_c and with lambdify
        kk, xx, bo = pbs.td.blocksimulation(N_steps)
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        if 0:
            plt.plot(kk*T, bo[bp_sum], label="Akrinor effect")
            plt.show()

        # output signal:

        y = bo[bp_sum]

        y_expected = np.array([66.5, 73.8])
        self.assertTrue(np.allclose(y[[199, 349]], y_expected, atol=.1))

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))


# #################################################################################################
#
# auxiliary functions
#
# #################################################################################################


def apx(x, x0, eps=1e-3):
    """
    express condition that x \approx x0
    """

    return (sp.Abs(x - x0) < eps)