import unittest
import pyblocksim as pbs
import numpy as np

from ipydex import IPS



class TestTD1(unittest.TestCase):
    def setUp(self):
        pbs.restart()

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

        kk, xx = pbs.td.blocksimulation(100)

        if 0:
            from matplotlib import pyplot as plt
            plt.plot(kk, xx)
            plt.plot(kk, xx[:,0]*0 + (1-np.exp(-1))*u_amplitude, "k-")
            plt.grid()
            plt.show()

        eval_k = int(u_step_time + T1/pbs.td.T)

        # evaluate 63% criterion
        self.assertAlmostEqual(xx[eval_k, 0], (1 - np.exp(-1))*u_amplitude)
