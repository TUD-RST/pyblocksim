import unittest
import pyblocksim as pbs

from ipydex import IPS



class TestTD1(unittest.TestCase):
    def setUp(self):
        pbs.restart()

    def test_block_construction1(self):

        block_class = pbs.td.new_TDBlock(2)
        dtPT1_1 = pbs.td.dtPT1()
        dtPT1_2 = pbs.td.dtPT1()

        self.assertEqual(pbs.td.dtPT1.instance_counter, 2)
        q = block_class()

        self.assertEqual(q.x2.name, "x4")

    def test_block_simulation1(self):
        u1_expr = pbs.td.td_step(pbs.td.k, 3, 10)
        dtPT1_1 = pbs.td.dtPT1(input1=u1_expr)
        dtPT1_2 = pbs.td.dtPT1(input1=dtPT1_1.Y)

        kk, xx = pbs.td.blocksimulation(30)
        from matplotlib import pyplot as plt
        plt.plot(kk, xx)
        plt.show()
