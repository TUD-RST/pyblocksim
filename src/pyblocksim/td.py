"""
This module contains a toolbox to construct, simulate and postprocess a graph of time discrete blocks
"""

from typing import List
import numpy as np
import sympy as sp

import symbtools as st

from ipydex import IPS


# discrete time

k = sp.Symbol("k")


class DataStore:
    instance = None
    def __init__(self):
        # ensure Singleton pattern
        assert self.instance is None
        self.instance = self

        self.initialize()

    def initialize(self):
        self.state_var_mapping = {}
        self.block_classes = {}
        self.block_instances = {}
        self.numbered_symbols = sp.numbered_symbols("x", start=1)

        self.global_rhs_expr = None
        self.all_state_vars = None
        self.rhs_func = None
        self.state_history = None

    def get_state_vars(self, n) -> List[sp.Symbol]:
        res = [next(self.numbered_symbols) for i in range(n)]
        return res

    def register_block(self, block: "TDBlock"):
        assert block.name not in self.state_var_mapping
        self.state_var_mapping[block.name] = block.state_vars

        assert block.name not in self.block_instances
        self.block_instances[block.name] = block


ds = DataStore()


class TDBlock:
    n_states = None
    instance_counter: int = None

    def __init__(self, name: str = None, input1=None, params=None):
        # increment the correct class attribute
        type(self).instance_counter += 1
        self.state_vars = ds.get_state_vars(self.n_states)

        if input1 is None:
            input1 = 0
        self.u1, = self.input_expr_list = [input1]

        # TODO: support multiple scalar inputs

        if params is None:
            params = {}
        self.params = params
        self.__dict__.update(params)

        # make each state variable available as `self.x1`, `self.x2`, ...
        for i, x_i in enumerate(self.state_vars, start=1):
            setattr(self, f"x{i}", x_i)

        if name is None:
            name = f"{type(self).__name__}__{self.instance_counter}"
        self.name = name

        ds.register_block(self)

    def output(self):
        """
        default output: first state

        (might be overridden by subclass)
        """
        return self.x1

    @property
    def Y(self):
        return self.output()


    def rhs(self, k: int, state: List) -> List:
        """
        Calculate the next state, based on the current state.

        This method has to be overridden by the subclasses
        """
        raise NotImplementedError


def new_TDBlock(n_states=None, suffix=None) -> type:
    assert isinstance(n_states, int) and n_states > 0

    if suffix is None:
        suffix = f"C{len(ds.block_classes) + 1}"

    name = f"TDBlock_{n_states}_{suffix}"

    assert name not in ds.block_classes

    dct = {"n_states": n_states, "instance_counter": 0}
    new_class = type(name, (TDBlock,), dct)
    ds.block_classes[name] = new_class
    return new_class

T = 0.1

####
class dtPT1(new_TDBlock(1)):

    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T1" in self.params

        x1,  = self.state_vars

        E1 = sp.exp(-T/self.T1)

        new_x1 = self.K*(1 - E1)*self.u1 + x1*E1

        return [new_x1]


class dtPT2(new_TDBlock(2)):

    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T1" in self.params
        assert "T2" in self.params

        if self.T1 != self.T2:
            raise NotImplementedError

        # calculate coefficients of time discrete transfer function

        x1, x2  = self.state_vars

        E1 = sp.exp(-T/self.T1)

        b1 = - 2*E1
        b2 = E1**2

        a1 = self.K*(1- E1 -  T/self.T1*E1)
        a2 = self.K*(E1**2 -E1 + T/self.T1 *E1)

        x2_new = a2*self.u1 - b2*x1
        x1_new = x2 + a1*self.u1 - b1*x1

        return [x1_new, x2_new]


class dtSigmoid(new_TDBlock(5)):

    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T_trans" in self.params
        assert "sens" in self.params

        # calculate coefficients of time discrete transfer function

        x1, x2, x_cntr, x_u_storage, x_debug  = self.state_vars
        x_u_storage_new = self.u1

        # determine a change of the input
        input_change = sp.Piecewise((1, sp.Abs(self.u1 - x_u_storage) > self.sens), (0, True))

        # counter
        delta_cntr = T/self.T_trans
        x_cntr_new =  sp.Piecewise(
            (delta_cntr, sp.Abs(self.u1 - x_u_storage) > self.sens),
            # note that expressions like 0 < x < 1 are not possible for sympy symbols
            (x_cntr + delta_cntr, (0 < x_cntr) &  (x_cntr<= 1)),
            (0, True),
        )

        T_fast = 2*T
        T1 = T_fast + self.T_trans*(1+40*sp.Abs(1-x_cntr)**10)/12
        x_debug_new = input_change
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        E1 = sp.exp(-T/T1)


        # this is a normal PT2 element but with dynamically adapted time constant
        b1 = - 2*E1
        b2 = E1**2

        a1 = self.K*(1- E1 -  T/T1*E1)
        a2 = self.K*(E1**2 -E1 + T/T1 *E1)

        x2_new = a2*self.u1 - b2*x1
        x1_new = x2 + a1*self.u1 - b1*x1

        return [x1_new, x2_new, x_cntr_new, x_u_storage_new, x_debug_new]


def blocksimulation(k_end):

    # generate equation system
    rhs_func = gen_global_rhs()
    initial_state = [0]*len(ds.all_state_vars)

    # solve equation system
    current_state = initial_state
    ds.state_history = [current_state]

    kk_num = np.arange(k_end)
    for k_num in kk_num[:-1]:
        current_state = rhs_func(k_num, *current_state)
        ds.state_history.append(current_state)

    ds.state_history = np.array(ds.state_history)

    return kk_num, ds.state_history


def gen_global_rhs():


    ds.global_rhs_expr = []
    ds.all_state_vars = []
    for block_name, state_vars in ds.state_var_mapping.items():
        block_instance: TDBlock = ds.block_instances[block_name]

        rhs_expr = list(block_instance.rhs(k, state_vars))

        ds.all_state_vars.extend(state_vars)
        ds.global_rhs_expr.extend(rhs_expr)

    ds.rhs_func = st.expr_to_func([k, *ds.all_state_vars], ds.global_rhs_expr, modules="numpy")

    return ds.rhs_func


def td_step(k, k_step, value1=1, value0=0):
    return sp.Piecewise((value0, k < k_step), (value1, True))
