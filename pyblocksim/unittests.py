# -*- coding: utf-8 -*-

import sys
import unittest
import importlib
import pickle
import os
import numpy as np
import pyblocksim as pbs
import inspect

# from ipHelp import IPS


"""
The main idea of this test suite is to run examples and compare
if the results are equal to manually checked reference data
(up to small numerical differences)
"""

# path to this file -> path to the examples dir
mod = sys.modules.get(__name__)
the_dir1 = os.path.dirname(os.path.abspath(mod.__file__))
the_dir2 = os.path.split(the_dir1)[0]
the_dir3 = os.path.join(the_dir2, "examples")

sys.path.append(the_dir3)

test_examples = {
                 1: 'example1',
                 2: 'example2',
                 3: 'example3',
                 4: 'example4',
                 5: 'example-hysteresis',
                 6: 'example06-output-derivative',
                 7: 'example07-delay',
                 }


def get_fname(name):
    py_version_str = "py{}".format(sys.version_info.major)
    return os.path.join(the_dir2, "testdata", py_version_str, name + ".pcl")


def generate_data(example_name):
    """
    run an example and save the results such that it can be compared to
    later on.
    """

    # delete all existing blocks
    pbs.restart()
    mod = importlib.import_module(example_name)

    # convert the block-output-dict
    bo = convert_dict_key_to_str(mod.bo)

    # from ipHelp import IPS
    # IPS()

    fname = get_fname(example_name)
    with open(fname, "wb") as pfile:
        pickle.dump(bo, pfile)
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

    if not len(thedict) == len(res):
        raise ValueError("The original dict contained at least two objects with the same name")
    return res


class TestInternals(unittest.TestCase):

    def setUp(self):
        pbs.restart()

    def tearDown(self):
        pass

    # uncomment this during debugging
    # @unittest.expectedFailure
    def test_debug_code_absent(self):
        """
        test whether there is some call to interactive IPython (leagacy from debugging)
        or some related import
        """

        # first load all relevant source files as list of lines in a dict
        srclines = {"core": inspect.getsourcelines(pbs.core)[0]}
        for _, example_name in test_examples.items():
            example_path = os.path.join(the_dir3, example_name)
            with open(example_path + ".py", 'r') as srcfile:
                src = srcfile.readlines()
            srclines[example_name] = src

        unittestfilename = inspect.getsourcefile(type(self))
        with open(unittestfilename, 'r') as srcfile:
            src = srcfile.readlines()
        srclines['unittest'] = src

        # now determine whether the lines are OK
        def filter_func(tup):
            idx, line = tup
            if line.strip().startswith('#'):
                return False
            str1 = 'I P S ()'.replace(' ', "")
            str2 = 'i p H e l p'.replace(' ', "")
            if str1 in line or str2 in line:
                return True

            return False

        for key, src in srclines.items():
            res = list(filter(filter_func, enumerate(src, 1)))

            # if there are any results, they are 2-tuples
            # we want to add information about the file (for good error msg)
            res = [(key, idx, line) for idx, line in res]
            self.assertEqual(res, [])

    def test_blocknames(self):

        u1, u2 = pbs.inputs("u1, u2")
        myblock = pbs.Blockfnc(3*u1)
        s = pbs.s
        blk2 = pbs.TFBlock((4*s + 2)/((s + 1)*(s + 2)), u1, name=u"äöü")

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

    def test_t00_bug(self):
        # bug: blocksimulation returns time array like r_[0, 0, dt, 2*dt, ...]
        # the doubl eoccurance of 0 is the problem
        mod = importlib.import_module(test_examples[1])

        dt1 = mod.t[1] - mod.t[0]
        dt2 = mod.t[2] - mod.t[1]
        self.assertEqual(mod.t[0], 0)
        self.assertEqual(dt1, dt2)

    def test_t00_bug2(self):
        # inputsignal should be evaluated correctly

        u1, = pbs.inputs('u1')  # external force and feedback
        meas1 = pbs.Blockfnc(u1)
        PT1 = pbs.TFBlock(1/(1 + pbs.s), u1)  #

        def u1fnc(t):
            if t == 0:
                return 0
            else:
                return 1

        t, states = pbs.blocksimulation(1, (u1, u1fnc))  # simulate 10 seconds

        bo = pbs.compute_block_ouptputs(states)
        uu1 = bo[meas1]
        self.assertEqual(uu1[0], 0)
        self.assertEqual(uu1[1], 1)
        self.assertEqual(uu1[2], 1)

    def test_block_output_dimension(self):
        # ensure that blockoutput is scalar
        mod = importlib.import_module(test_examples[1])

        self.assertEqual(len(list(mod.bo.values())[0].shape), 1)

    def test_ringbuffer(self):

        bufferlength = 7
        rb = pbs.RingBuffer(bufferlength)
        N = 50
        arr_in = np.zeros(N)
        arr_out = np.zeros(N)
        ii = np.arange(N)
        for i in ii:
            x = i**2 + 500
            arr_in[i] = x
            arr_out[i] = rb.read()
            rb.write_and_step(x)

        if 0:
            import matplotlib.pyplot as plt
            plt.plot(ii, arr_in)
            plt.plot(ii, arr_out)
            plt.show()

        self.assertTrue(np.allclose(arr_in[:-bufferlength], arr_out[bufferlength:]))


class TestExamples(unittest.TestCase):

    def setUp(self):
        pbs.restart()
        pass

    def tearDown(self):
        pass

    def specific_example_test(self, K):
        mod = importlib.import_module(test_examples[K])
        bo_ref = load_data(K)

        # bo_ref = convert_dict_key_to_str(bo_ref)
        if K != 6:
            return
        mod.bo = convert_dict_key_to_str(mod.bo)

        self.assertEqual(set(mod.bo.keys()), set(bo_ref.keys()))
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

    def test_example_output_deriv(self):
        self.specific_example_test(6)

    def test_example_delay(self):
        self.specific_example_test(7)


if __name__ == '__main__':

    if 'generate_data' in sys.argv:
        for k, v in test_examples.items():
            generate_data(v)
    elif 'gd07' in sys.argv:
            generate_data(test_examples[7])
    else:
        unittest.main()

