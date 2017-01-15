# -*- coding: utf-8 -*-

import sys
import unittest
import importlib
import pickle
import os
import numpy as np
import pyblocksim as pbs

from ipHelp import IPS


"""
The main idea of this test suite is to run examples and compare
if the results are equal to manually checked reference data
(up to small numerical differences)
"""

test_examples = {1: 'example1',
                 2: 'example2',
                 3: 'example3',
                 4: 'example4',
                 5: 'example-hysteresis',
                 }


def get_fname(name):
    return os.path.join("testdata", name + ".pcl")


def generate_data():
    """
    run all examples and save the results such that it can be compared to
    them later on.
    """
    for k, v in test_examples.items():
        mod = importlib.import_module(v)

        fname = get_fname(v)
        with open(fname, "wb") as pfile:
            pickle.dump(mod.bo, pfile)
        print(fname, " written.")


def load_data(key):
    name = test_examples[key]
    fname = get_fname(name)
    with open(fname, "rb") as pfile:
        bo = pickle.load(pfile)
    return bo


def convert_dict_key_to_str(thedict):
    """
    The dicts which contain the data use python objects as keys.
    These are unique for each python session. To use one key for both
    dicts we better use names.
    """
    res = {}
    for k, v in thedict.items():
        res[k.name] = v
    return res


class TestInternals(unittest.TestCase):

    def setUp(self):
        pbs.restart()

    def tearDown(self):
        pass

    def test_blocknames(self):

        u1, u2 = pbs.inputs("u1, u2")
        myblock = pbs.Blockfnc(3*u1)

        tmp_block = myblock
        # this should generate a new name
        myblock = pbs.Blockfnc(3*u1)
        self.assertTrue(myblock.name != tmp_block.name)

        myblock1 = pbs.Blockfnc(3*u1, 'block1')
        # this should generate a warning due to reuse of name 'block1'
        with pbs.warnings.catch_warnings(record=True) as cm:
            myblock1 = pbs.Blockfnc(3*u1, 'block1')

        self.assertEqual(len(cm), 1)
        self.assertTrue('block1' in str(cm[0].message))


class TestExamples(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def specific_example_test(self, K):
        mod = importlib.import_module(test_examples[K])
        bo_ref = load_data(K)

        bo_ref = convert_dict_key_to_str(bo_ref)
        mod.bo = convert_dict_key_to_str(mod.bo)
        IPS()

        self.assertEquals(mod.bo.keys(), bo_ref.keys())
        for k in mod.bo.keys():
            arr = mod.bo[k]
            arr_ref = bo_ref[k]
            self.assertTrue( all(np.isclose(arr, arr_ref)) )

    def test_example1(self):
        self.specific_example_test(1)

    def test_example2(self):
        self.specific_example_test(2)

    def test_example3(self):
        self.specific_example_test(3)

    def test_example4(self):
        self.specific_example_test(4)

    def test_example_hyst(self):
        self.specific_example_test(5)


if __name__ == '__main__':

    if 'generate_data' in sys.argv:
        generate_data()
    else:
        unittest.main()

