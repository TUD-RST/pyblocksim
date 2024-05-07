"""
This module contains a toolbox to construct, simulate and postprocess a graph of time discrete blocks
"""

from typing import List
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function

import symbtools as st

from ipydex import IPS


# discrete time

k = sp.Symbol("k")

# this will be replaced by k*T in the final equations
t = sp.Symbol("t")


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

        # to calculate and store the output
        self.output_fnc = None
        self.output_res = None

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

    def __repr__(self):
        return type(self).__name__ + ":" + self.name


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


class StaticBlock(TDBlock):

    n_states = 0
    instance_counter = 0

    def __init__(self, name: str = None, output_expr: sp.Expr = None):
        super().__init__(name=name, input1=None, params=None)
        self.output_expr = output_expr

    def rhs(self, k: int, state: List) -> List:
        return state

    def output(self):
        return self.output_expr


# discrete step time
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


class dtPT2Euler(new_TDBlock(2)):
    """
    PT2 element discretized via Eulers forward method
    """


    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T1" in self.params
        assert "T2" in self.params

        if self.T1 != self.T2:
            raise NotImplementedError

        # calculate coefficients of time discrete transfer function

        x1, x2  = self.state_vars

        x1_new = x1 + T*x2
        x2_new = x2 + T*(1/(self.T1*self.T2)*(-(self.T1 + self.T2)*x2 - x1 + self.u1))

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

        # this will reach 0 before x_cntr will reach 1
        count_down = limit(1-1.2*x_cntr, xmin=0, xmax=1, ymin=0, ymax=1)

        T1 = T_fast + .6*self.T_trans*(1+40*count_down**10)/12
        x_debug_new = input_change
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        p12 = 0.6

        phase2 = limit(x_cntr, xmin=p12, xmax=1, ymin=0, ymax=1)
        phase1 = 1 - phase2

        # PT2 Element based on Euler forward approximation
        x1_new = sum((
            x1,
            (T*x2)*phase1,    # ordinary PT2 part
            (T/T_fast*(self.K*self.u1 - x1))*phase2,
        ))

        # at the very end we want x1 == K*u1
        x1_new = sp.Piecewise((x1_new, x_cntr<= 1), (self.K*self.u1, True))


        # x2 should go to zero at the end of transition
        x2_new = sum((
            x2*phase1,
            T*(1/(T1*T1)*(-(T1 + T1)*x2 - x1 + self.K*self.u1))*phase1,

        ))

        return [x1_new, x2_new, x_cntr_new, x_u_storage_new, x_debug_new]


class dtDirectionSensitiveSigmoid(new_TDBlock(5)):
    """
    This Sigmoid behaves differently for positive and negative input steps
    """

    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T_trans_pos" in self.params # overall counter time
        assert "T_trans_neg" in self.params
        assert "sens" in self.params

        # fraction of overall counter time that is used for waiting
        f_wait_pos = getattr(self, "f_wait_pos", 0)
        f_wait_neg = getattr(self, "f_wait_neg", 0)

        assert 0 <= f_wait_pos <= 1
        assert 0 <= f_wait_neg <= 1

        x1, x2, x_cntr, x_u_storage, x_debug  = self.state_vars
        x_u_storage_new = self.u1

        # determine a change of the input
        input_change = sp.Piecewise((1, sp.Abs(self.u1 - x_u_storage) > self.sens), (0, True))

        # counter
        pos_delta_cntr = T/self.T_trans_pos
        neg_delta_cntr = -T/self.T_trans_neg
        x_cntr_new =  sp.Piecewise(
            (pos_delta_cntr, self.u1 - x_u_storage > self.sens),
            (neg_delta_cntr, self.u1 - x_u_storage < - self.sens),
            # note that expressions like 0 < x < 1 are not possible for sympy symbols
            (x_cntr + pos_delta_cntr, (0 < x_cntr) & (x_cntr<= 1)),
            (x_cntr + neg_delta_cntr, (0 < -x_cntr) & (-x_cntr<= 1)),
            (0, True),
        )

        # implement the waiting
        # effective waiting fraction
        f_wait = sp.Piecewise((f_wait_neg, x_cntr < 0), (f_wait_pos, x_cntr > 0), (0, True))

        # effective counter (reaching the goal early by intention)
        x_cntr_eff = limit(sp.Abs(x_cntr), xmin=f_wait, xmax=.95, ymin=0, ymax=1)*sign(x_cntr)

        T_fast = 2*T

        # this will reach 0 before |x_cntr_eff| will reach 1
        # count_down = limit(1-1.2*sp.Abs(x_cntr_eff), xmin=0, xmax=1, ymin=0, ymax=1)
        count_down = 1 - sp.Abs(x_cntr_eff)


        T_trans = sp.Piecewise((self.T_trans_neg, x_cntr < 0), (self.T_trans_pos, x_cntr > 0), (0, True) )

        # !! self muss raus
        T1 = T_fast + .6*self.T_trans_pos*(1+40*count_down**10)/12

        p12 = 0.6

        phase0 = sp.Piecewise((1, sp.Abs(x_cntr_eff) <= 1e-4), (0, True))
        phase2 = limit(sp.Abs(x_cntr_eff), xmin=p12, xmax=1, ymin=0, ymax=1)*(1-phase0)
        phase1 = (1 - phase2)*(1-phase0)


        x_debug_new = x_cntr_eff#  phase0*10 + phase1
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        # PT2 Element based on Euler forward approximation
        x1_new = sum((
            x1,
            (T*x2)*phase1,    # ordinary PT2 part
            (T/T_fast*(self.K*self.u1 - x1))*phase2,  # fast PT1-convergence towards the stationary value
        ))

        # at the very end we want x1 == K*u1 (exactly)
        x1_new = sp.Piecewise((x1_new, sp.Abs(x_cntr)<= 1), (self.K*self.u1, True))


        # x2 should go to zero at the end of transition
        x2_new = sum((
            x2*phase1,
            T*(1/(T1*T1)*(-(T1 + T1)*x2 - x1 + self.K*self.u1))*phase1,

        ))

        return [x1_new, x2_new, x_cntr_new, x_u_storage_new, x_debug_new]


class dtRelaxoBlock(new_TDBlock(5)):
    """
    This block models relaxometrics
    """

    def rhs(self, k: int, state: List) -> List:

        assert "sens" in self.params
        assert "K" in self.params
        assert "slope" in self.params
        # assumption: next nonzero input not in sigmoid phase

        x1, x2, x_cntr, x4_buffer, x_debug  = self.state_vars

        # time for sigmoid phase
        self.T_phase1 = 5


        # counter
        pos_delta_cntr = T/self.T_phase1
        x_cntr_new =  sp.Piecewise(
            (pos_delta_cntr, self.u1 > 0),
            (x_cntr + pos_delta_cntr, (0 < x_cntr) & (x_cntr<= 1)),
            (0, True),
        )

        sigmoid_phase_cond = (self.u1 > 0) | ((0 < x_cntr) & (x_cntr<= 1))

        # if sigmoid phase is over (counter reached 1) x4 := x1
        x4_buffer_new = sp.Piecewise((x4_buffer + self.K*self.u1, sigmoid_phase_cond), (x1, True))

        # effective counter (here: same as normal)
        x_cntr_eff = x_cntr

        T_fast = 2*T
        count_down = 1 - sp.Abs(x_cntr_eff)


        T_trans = sp.Piecewise((self.T_phase1, x_cntr > 0), (0, True) )

        T1 = T_fast + .6*T_trans*(1+40*count_down**10)/12

        p12 = 0.6


        counter_active = sp.Piecewise((1, x_cntr > 0), (0, True))

        phase0 = 0# sp.Piecewise((1, sp.Abs(x_cntr_eff) <= 1e-4), (0, True))
        phase2 = 0# limit(sp.Abs(x_cntr_eff), xmin=p12, xmax=1, ymin=0, ymax=1)*(1-phase0)
        phase1 = counter_active


        x_debug_new = x_cntr_eff#  phase0*10 + phase1
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        # PT2 Element based on Euler forward approximation (but here x4 is the "input")

        v = x4_buffer

        x1_new = sum((
            x1,
            (T*x2)*counter_active,   # ordinary PT2 part

            # decay with constant rate
            (self.slope)*sp.Piecewise((1, x1 > 0), (0, True))*(1-counter_active)  # assumption: slope < 0
        ))

        # at the very end we want x1 == K*u1 (exactly)
        x1_new = sp.Piecewise((x1_new, sp.Abs(x_cntr)<= 1), (v, True))

        # x2 should go to zero at the end of transition
        x2_new = sum((
            x2*phase1,
            T*(1/(T1*T1)*(-(T1 + T1)*x2 - x1 + v))*phase1,
        ))

        return [x1_new, x2_new, x_cntr_new, x4_buffer_new, x_debug_new]

    def output(self):
        return 1 - self.x1


class dtSulfenta(new_TDBlock(5)):
    """
    This block models pain suppression with Sulfentanil
    """

    def rhs(self, k: int, state: List) -> List:

        assert "rise_time" in self.params
        assert "down_slope" in self.params
        assert self.down_slope < 0

        # how long effect stays constant (dependent on input)
        assert "active_time_coeff" in self.params
        assert "dose_gain" in self.params

        # assumption: next nonzero input not in rising or const phase

        x1, x2_target_effect, x3_cntr, x4_slope, x_debug  = self.state_vars

        # value by which the counter is increased in every step
        # after N = rise_time/T steps the counter reaches 1
        delta_cntr1 = T/self.rise_time

        eps = 1e-8 # prevent ZeroDivisionError when calculating unused intermediate result
        delta_cntr2 = sp.Piecewise(
            (T/(self.active_time_coeff*(x2_target_effect/self.dose_gain + eps)), x2_target_effect > 0),
            (0, True)
        )
        x3_cntr_new =  sp.Piecewise(
            (delta_cntr1, self.u1 > 0),
            (x3_cntr + delta_cntr1, (0 < x3_cntr) & (x3_cntr<= 1)),
            # after counter reached 1 -> count from 1 to 2
            (x3_cntr + delta_cntr2, (1 < x3_cntr) & (x3_cntr<= 2)),
            (x3_cntr, (2 < x3_cntr) & (x3_cntr<= 3) & (x1 > 0)),
            (0, True),
        )


        # save uninfluenced target effect for input dose if it is unequal zero
        # "uninfluenced" means as if it was starting from zero
        # -> this yields the correct slope and active time
        # however the real value of x1 might be higher, if the 2nd input dose comes for x1 > 0
        # then x1 is simply increased by the slope
        x2_target_effect_new = sp.Piecewise(
            (x1*0+ self.u1*self.dose_gain, self.u1 > 0),
            (x2_target_effect, True),
        )

        x4_slope_new = sp.Piecewise(
            (x2_target_effect_new/self.rise_time*T, self.u1 > 0),
            (x4_slope, True),
        )

        x1_new = sp.Piecewise(
            (x1 + x4_slope_new, (0 < x3_cntr) & (x3_cntr<= 1)),
            (x1, (1 < x3_cntr) & (x3_cntr<= 2)),
            (x1 + T*self.down_slope, (2 < x3_cntr) & (x3_cntr<= 3) & (x1 >  T*self.down_slope)),
            (0, True),
        )

        x_debug_new = 0

        res = [x1_new, x2_target_effect_new, x3_cntr_new, x4_slope_new, x_debug_new]
        return res


    def output(self):
        return self.x1


def tmp_eq_fnc(x4, i, res):
    if x4 == i:
        return res
    else:
        return 0

eq_fnc = implemented_function(f"eq_fnc", tmp_eq_fnc)


N_acrinor_counters = 3
class dtAcrinor(new_TDBlock(5 + N_acrinor_counters*2)):
    """
    This block models blood pressure increase due to Acrinor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.non_counter_states = self.state_vars[:len(self.state_vars)-2*N_acrinor_counters]
        self.counter_states = self.state_vars[len(self.state_vars)-2*N_acrinor_counters:]
        self.n_counters = N_acrinor_counters

    def rhs(self, k: int, state: List) -> List:

        # time to exponentially rise 75%
        assert "T_75" in self.params
        assert "T_plateau" in self.params
        assert "body_mass" in self.params
        assert "down_slope" in self.params
        assert self.down_slope < 0

        # gain in mmHg/(ml/kgG)
        # TODO: this has to be calculated as percentage (see red Text in pptx)
        assert "dose_gain" in self.params

        # assumptions:
        # next nonzero input not in rising phase
        # if next nonzero input in plateau or down_slope phase this has to be handled differently (second element)

        x1, x2_integrator, x3_PT1, x4_counter_idx, x5_debug  = self.non_counter_states

        # conventional time constant for exponential rising
        T1 = self.T_75/np.log(4)


        # TODO: u1 value
        # absolute_map_increase must be calculated according characteristic curve and current MAP
        absolute_map_increase = self.u1

        # functions for handling the counters
        def counter_func_imp(counter_state, counter_k_start, k, counter_index_state, i, initial_value):
            # calculate the new counter state
            if counter_index_state == i and initial_value > 0:
                # if counter state is newly loaded it should be zero before
                assert counter_state == 0
                return initial_value

            if k >= counter_k_start:
                res = counter_state + self.down_slope*0.01
                if res < 0:
                    res = 0
                return res
            return counter_state

        counter_func = implemented_function(f"counter_func", counter_func_imp)


        def counter_start_func_imp(counter_k_start, k, counter_index_state, i, initial_value):

            if counter_index_state == i and initial_value > 0:
                # the counter k_start should be set
                return k + self.T_plateau/T

            # change nothing
            return counter_k_start

        counter_start_func = implemented_function(f"counter_start_func", counter_start_func_imp)

        # this acts as the integrator
        counter_sum = 0
        for i in range(self.n_counters):

            # counter_value for index i
            self.counter_states[2*i] = counter_func(
                self.counter_states[2*i], self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

            counter_sum += self.counter_states[2*i]

            # k_start value for index i
            self.counter_states[2*i + 1] = counter_start_func(
                self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

        new_counter_states = self.counter_states

        e1 = np.exp(-T/T1)
        countdown = 0

        x1_new = x3_PT1 - countdown
        x2_new = 0 # x2_integrator + absolute_map_increase
        x3_new = e1*x3_PT1 + 1*(1-e1)*counter_sum

        # increase the counter index for every nonzero input
        x4_new = x4_counter_idx + sp.Piecewise((1, absolute_map_increase >0), (0, True)) % self.n_counters
        x5_debug_new = 0

        res = [x1_new, x2_new, x3_new, x4_new, x5_debug_new] + new_counter_states
        return res


    def output(self):
        return self.x1



def limit(x, xmin=0, xmax=1, ymin=0, ymax=1):

    dx = xmax - xmin
    dy = ymax - ymin
    m = dy/dx

    new_x_expr = ymin + (x - xmin)*m

    return sp.Piecewise((ymin, x < xmin), (new_x_expr, x < xmax), (ymax, True))


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

    block_outputs = compute_block_outputs(kk_num, ds.state_history)

    return kk_num, ds.state_history, block_outputs


def gen_global_rhs():

    ds.global_rhs_expr = []
    ds.all_state_vars = []

    rplmts = [(t, k*T)]

    for block_name, state_vars in ds.state_var_mapping.items():
        block_instance: TDBlock = ds.block_instances[block_name]

        rhs_expr = list(block_instance.rhs(k, state_vars))

        ds.all_state_vars.extend(state_vars)

        rhs_expr2 = []
        for elt in rhs_expr:
            try:
                rhs_expr2.append(elt.subs(rplmts))
            except AttributeError:
                assert sp.Number(elt) == elt
                rhs_expr2.append(elt)
                continue

        ds.global_rhs_expr.extend(rhs_expr2)

        if hasattr(block_instance, "output_expr"):
            block_instance.output_expr = block_instance.output_expr.subs(rplmts)

    assert len(ds.global_rhs_expr) == len(ds.all_state_vars)
    ds.rhs_func = st.expr_to_func([k, *ds.all_state_vars], ds.global_rhs_expr, modules="numpy")

    return ds.rhs_func


def td_step(k, k_step, value1=1, value0=0):
    return sp.Piecewise((value0, k < k_step), (value1, True))

def sign(x):
    return sp.Piecewise((-1, x < 0), (0, x == 0), (1, True))


def compute_block_outputs(kk, xx) -> dict:
    """
    """

    # get functions for outputs

    block_outputs = {}
    for block_name in ds.state_var_mapping.keys():
        block: TDBlock = ds.block_instances[block_name]

        if block.output_fnc is None:
            block.output_fnc = st.expr_to_func([k, *ds.all_state_vars], block.output())
        block.output_res = block_outputs[block] = block.output_fnc(kk, *xx.T)

    return block_outputs
