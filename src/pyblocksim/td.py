"""
This module contains a toolbox to construct, simulate and postprocess a graph of time discrete blocks
"""

from typing import List
from string import Template
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function

import symbtools as st

from ipydex import IPS


# discrete time

k = sp.Symbol("k")

# this will be replaced by k*T in the final equations
t = sp.Symbol("t")

loop_symbol_iterator = sp.numbered_symbols("loop")
loop_mappings = {}


def perform_sympy_monkey_patch_for__is_constant():
    """
    This monkey patch is necessary for the following reason:
    sp.lambdify calls a python printer. Deep down within the printer it must be determined
    whether an expression is constant or not. Sympy does this by substituting some numbers
    into the expression and looks for changes. However, we use implemented functions which
    raise exceptions for some combinations of arguments, but we still want to lambdify them.

    Solution replace the original .is_constant-method with a wrapper which handles our
    special case.
    """

    if getattr(sp.Expr, "_orig_is_constant", None) is None:
        sp.Expr._orig_is_constant = sp.Expr.is_constant

    def new_is_constant(self, *args, **kwargs) -> bool:
        for func_atoms in self.atoms(sp.core.function.AppliedUndef):
            if func_atoms.atoms(sp.Symbol):
                return False
        return self._orig_is_constant(*args, **kwargs)

    sp.Expr.is_constant = new_is_constant

perform_sympy_monkey_patch_for__is_constant()

# abbreviation for the equality operator (needed in some Piecewise definitions)
eq = sp.Equality


# TODO: write unittest for this mechanism
def get_loop_symbol():
    ls = next(loop_symbol_iterator)
    loop_mappings[ls] = None
    return ls


def set_loop_symbol(ls, expr):
    assert loop_mappings[ls] is None
    loop_mappings[ls] = expr


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

    def __init__(self, name: str = None, input1=None, params=None, **kwargs):
        # increment the correct class attribute
        type(self).instance_counter += 1
        self.state_vars = ds.get_state_vars(self.n_states)

        # to calculate and store the output
        self.output_fnc = None
        self.output_res = None

        if input1 is None:
            input1 = 0
        self.u1, = self.input_expr_list = [input1]

        i = 1
        for key, value in kwargs.items():
            i += 1
            # assume key like input3
            assert key.startswith("input")
            assert i == int(key.replace("input", ""))
            setattr(self, f"u{i}", value)
            self.input_expr_list.append(value)

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

        # store implemented functions (e.g. for counter handling)
        self._implemented_functions = {}

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


class dtSufenta(new_TDBlock(5)):
    """
    This block models pain suppression with Sufentanil
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

class dtDelay1(new_TDBlock(1)):
    def rhs(self, k: int, state: List) -> List:
        x1, = self.state_vars
        new_x1 = self.u1
        return [new_x1]

    def output(self):
        return self.x1

def debug_func_imp(cond, *args, **kwargs):
    print(args)
    print(kwargs)
    return 0

debug_func = implemented_function(f"debug_func", debug_func_imp)


class MaxBlockMixin:
    """
    Allows to calculate the maximum of a length3-sequence of expressions
    """
    def _define_max3_func(self):
        cached_func = self._implemented_functions.get("max3_func")
        if cached_func is not None:
            return cached_func

        def max3_func_imp(a, b, c):
            return max(a, b, c)

        max3_func = implemented_function("max3_func", max3_func_imp)

        # the following is necessary for fast implementation
        max3_func.c_implementation = Template("""
            double max3_func(double a, double b, double c) {
                double v1, v2;
                v1 = fmax(a, b);
                v2 = fmax(b, c);
                return fmax(v1, v2)
            }
        """).substitute()

        self._implemented_functions["max3_func"] = max3_func
        return max3_func


class CounterBlockMixin:
    def _define_counter_func_1state(self):
        """
        This is for a 1-state counter (it is started in the rhs function) and just counts down
        """

        cached_func = self._implemented_functions.get("counter_func_2state")
        if cached_func is not None:
            return cached_func

        def counter_func_1state_imp(counter_state, counter_index_state, i, initial_value):
            """
            :param counter_state:   float; current value of the counter
            :param counter_index_state:
                                    int: index which counter should be activated next (was not active)
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """
            # check if the counter-loop (i) is considering the counter which should be activated next
            if counter_index_state == i and initial_value > 0:
                # activate this counter

                assert counter_state == 0
                return initial_value
            if counter_state > 0:
                return counter_state - 1

            return 0
        # convert the python-function into a applicable sympy function
        counter_func_1state = implemented_function("counter_func_1state", counter_func_1state_imp)

        # the following is necessary for fast implementation
        counter_func_1state.c_implementation = Template("""
            double counter_func_1state(double counter_state, double counter_index_state, double i, double initial_value) {
                double result;
                double down_slope = $down_slope;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // assign the initial value to the counter_state
                    return initial_value;
                }

                // check if the counter (i) is currently running
                if (counter_state > 0) {

                    // counter is assumed to be counting down, thus down_slope is < 0
                    result = counter_state + down_slope;
                    if (result < 0) {
                        result = 0;
                    }
                    return result;
                }

                // if the counter is not running, return 0
                return 0;
            }
        """).substitute(down_slope=-1)

        self._implemented_functions["counter_func_1state"] = counter_func_1state
        return counter_func_1state

    def _define_counter_func_2state(self):
        """
        This is for a 2-state counter (which starts not now but in somewhere in the future)
        """

        cached_func = self._implemented_functions.get("counter_func_2state")
        if cached_func is not None:
            return cached_func

        def counter_func_2state_imp(counter_state, counter_k_start, k, counter_index_state, i, initial_value):
            """
            :param counter_state:   float; current value of the counter
            :param counter_k_start: int; time step index when this counter started
            :param k:               int; current time step index
            :param counter_index_state:
                                    int: index which counter should be activated next (was not yet active)
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """
            # check if the counter-loop (i) is considering the counter which should be activated next
            if counter_index_state == i and initial_value > 0:

                # if counter state is newly loaded it should be zero before
                # print(f"{k=}, new iv: {initial_value}")
                assert counter_state == 0

                # assign the initial value to the counter_state
                return initial_value

            # check if the counter (i) is currently running
            if (counter_state > 0) and (k >= counter_k_start):

                # counter is assumed to be counting down, thus down_slope is < 0
                res = counter_state + self.down_slope
                if res < 0:
                    res = 0
                return res

            # if the counter is not running, do not change the state
            return counter_state

        # convert the python-function into a applicable sympy function
        counter_func_2state = implemented_function("counter_func_2state", counter_func_2state_imp)

        # the following is necessary for fast implementation
        counter_func_2state.c_implementation = Template("""
            double counter_func_2state(double counter_state, double counter_k_start, double k, double counter_index_state, double i, double initial_value) {
                double result;
                double down_slope = $down_slope;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // assign the initial value to the counter_state
                    return initial_value;
                }

                // check if the counter (i) is currently running
                if ((counter_state > 0) && (k >= counter_k_start)) {

                    // counter is assumed to be counting down, thus down_slope is < 0
                    result = counter_state + down_slope;
                    if (result < 0) {
                        result = 0;
                    }
                    return result;
                }

                // if the counter is not running, do not change the state
                return counter_state;
            }
        """).substitute(down_slope=self.down_slope)

        self._implemented_functions["counter_func_2state"] = counter_func_2state
        return counter_func_2state

    def _define_counter_start_func_2state(self, delta_k):
        ":param delta_k: the amount of time_steps in the future when the counter will start"

        cached_func = self._implemented_functions.get("counter_start_func_2state")
        if cached_func is not None:
            return cached_func


        def counter_start_func_2state_imp(counter_k_start, k, counter_index_state, i, initial_value):
            """
            :param counter_k_start: int; time step index when this counter started
            :param k:               int; current time step index
            :param counter_index_state:
                                    int: index which counter should be activated next
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """

            if counter_index_state == i and initial_value > 0:
                # the counter k_start should be set
                return k + delta_k

            # change nothing
            return counter_k_start

        counter_start_func_2state = implemented_function(f"counter_start_func_2state", counter_start_func_2state_imp)
        counter_start_func_2state.c_implementation = Template("""

            double counter_start_func_2state(double counter_k_start, double k, double counter_index_state, double i, double initial_value) {
                double delta_k = $delta_k;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // the counter k_start should be set
                    return k + delta_k;
                }

                // change nothing
                return counter_k_start;
            }


        """).substitute(delta_k=delta_k)

        self._implemented_functions["counter_start_func_2state"] = counter_start_func_2state

        return counter_start_func_2state


# This determines how many overlapping Akrinor bolus doses can be modelled
# Should be increased to 10
N_akrinor_counters = 3
class dtAkrinor(new_TDBlock(5 + N_akrinor_counters*2), CounterBlockMixin):
    """
    This block models blood pressure increase due to Akrinor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.non_counter_states = self.state_vars[:len(self.state_vars)-2*N_akrinor_counters]
        self.counter_state_vars = self.state_vars[len(self.state_vars)-2*N_akrinor_counters:]

        self.counter_states = self.counter_state_vars
        self.n_counters = N_akrinor_counters

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

        # assume self.u2 is the current system MAP
        assert isinstance(self.u2, sp.Expr)

        # assumptions:
        # next nonzero input not in rising phase
        # if next nonzero input in plateau or down_slope phase this has to be handled differently (second element)

        x1, x2_integrator, x3_PT1, x4_counter_idx, x5_debug  = self.non_counter_states

        # conventional time constant for exponential rising
        T1 = self.T_75/np.log(4)

        # absolute_map_increase will be the plateau value of the curve
        # absolute_map_increase must be calculated according characteristic curve and current MAP
        # u1: dose of current bolus, u2: current MAP
        absolute_map_increase = self.u1*self.dose_gain/self.body_mass*self.u2

        """
        The counter mechanism works like this:

        - Every counter is associated with two scalar states
        - c0 is associated with self.counter_states[0] and self.counter_states[1]
        - self.counter_states[0]
        - self.counter_states[1]
        - all counters start inactive; if c0 gets activated self.counter_states[0] gets a nonzero value
        - in every time step: all counter_states have to be updated, because of the paradigm:
            new_total_state := state_func(current_total_state)
        - if in time step k the input (`initial_value`) is non-zero the counter which is associated with
            counter_index_state gets prepared. More precisely two things happen (assuming c0 is the one):
            - `counter_func_2state` -> load initial value into self.counter_states[0]
            - `counter_start_func_2state` -> load the index into self.counter_states[1] at which the counter
                actually will start to count down (after T_plateau is over)
        """
        # create/restore functions for handling the counters
        counter_func_2state = self._define_counter_func_2state()

        delta_k = int(self.T_plateau/T)
        counter_start_func_2state = self._define_counter_start_func_2state(delta_k=delta_k)

        # this acts as the integrator
        counter_sum = 0

        new_counter_states = [None]*len(self.counter_states)
        # the counter-loop
        for i in range(self.n_counters):

            # counter_value for index i
            new_counter_states[2*i] = counter_func_2state(
                self.counter_states[2*i], self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

            counter_sum += new_counter_states[2*i]

            # k_start value for index i
            new_counter_states[2*i + 1] = counter_start_func_2state(
                self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

        # T1: time constant of PT1 element, e1: factor for time discrete PT1 element
        e1 = np.exp(-T/T1)

        # old: PT1-like decreasing
        # x1_new = x3_PT1

        # new: PT1-like increasing, linear decreasing
        # (this requires the monkey-patch for is_constant (see above))
        x1_new = sp.Piecewise((x3_PT1, x3_PT1 < counter_sum), (counter_sum, True))

        # currently not used
        x2_new = 0 # x2_integrator + absolute_map_increase

        # counter_sum serves as input signal with stepwise increase and linear decrease
        x3_new = e1*x3_PT1 + 1*(1-e1)*counter_sum

        # increase the counter index for every nonzero input, but start at 0 again
        # if all counters have been used
        x4_new = (x4_counter_idx + sp.Piecewise((1, absolute_map_increase >0), (0, True))) % self.n_counters
        x5_debug_new = 0 # debug_func(self.u1 > 0, k, self.u2, "k,u2")

        res = [x1_new, x2_new, x3_new, x4_new, x5_debug_new] + new_counter_states
        return res

    def output(self):
        return self.x1


# for backward compatibility (Notebooks with Typos)
dtSulfenta = dtSufenta
dtAcrinor = dtAkrinor


N_propofol_counters = 3
class dtPropofolBolus(new_TDBlock(5 + 2*N_propofol_counters), CounterBlockMixin, MaxBlockMixin):
    """
    This block models blood pressure increase due to Propofol bolus doses.
    It uses 1state counters. Each counter, is followed by an associated amplitude value.
    (also considered part of the counter state components ("counter states"))

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.non_counter_states = self.state_vars[:len(self.state_vars)-2*N_propofol_counters]
        self.counter_states = self.state_vars[len(self.state_vars)-2*N_propofol_counters:]

        self.n_counters = N_propofol_counters

        # Note: counter is used both for effect and sensitivity
        self.T_counter = 6

        self.propofol_bolus_sensitivity_dynamics_np = st.expr_to_func(
            t, self.propofol_bolus_sensitivity_dynamics(t), modules="numpy"
        )

        dose = sp.Symbol("dose")
        self.propofol_bolus_static_values_np = st.expr_to_func(
            dose, self.propofol_bolus_static_values(dose), modules="numpy"
        )

        self.bp_effect_dynamics_expr = self._generate_effect_dynamics_expr()

    def _generate_effect_dynamics_expr(self):
        """
        Generate a piecewise defined polynomial like: _/‾‾\_ (amplitude 1)
        """
        Ta = 2
        Tb = 4
        Tc = 6

        # rising part
        poly1 = st.condition_poly(t, (0, 0, 0, 0), (Ta, 1, 0, 0)) ##:

        # falling part
        poly2 = st.condition_poly(t, (Tb, 1, 0, 0), (Tc, 0, 0, 0)) ##:

        effect_dynamics_expr = sp.Piecewise(
            (0, t < 0), (poly1, t <= Ta), (1, t <= Tb), (poly2, t <= Tc), (0, True)
        )
        return effect_dynamics_expr


    def rhs(self, k: int, state: List) -> List:

        x1_bp_effect, x2_sensitivity, x3, x4_counter_idx, x5_debug  = self.non_counter_states

        # the counter should start immediately -> 1 state version
        # create/restore functions for handling the counters
        counter_func_1state = self._define_counter_func_1state()


        # TODO: make this better parameterizable
        counter_max_val = self.T_counter/T
        counter_target_val = sp.Piecewise((counter_max_val, self.u1 > 0), (0, True))

        new_counter_states = [None]*len(self.counter_states)
        partial_sensitivities = [None]*self.n_counters
        partial_bp_effects = [None]*self.n_counters

        # amplitude_func = self._define_amplitude_func() # (self.u1, *self.counter_states)

        # the counter-loop
        # Explanation: For counter index i `self.counter_states[2*i]` is the counter value
        # and `self.counter_states[2*i + 1]` is the associated amplitude value.
        # The amplitude depends on current input (`self.u1`) and current sensitivity (`x2`)
        new_amplitude = self.propofol_bolus_static_values(self.u1 * x2_sensitivity)

        # it is mostly 0, except when there is a nonzero input

        for i in range(self.n_counters):

            # counter_value for index i
            counter_i = new_counter_states[2*i] = counter_func_1state(
                self.counter_states[2*i], x4_counter_idx, i, counter_target_val
            )

            counter_time = sp.Piecewise(((counter_max_val - counter_i)*T, counter_i > 0), (0, True))
            partial_sensitivities[i] = self.propofol_bolus_sensitivity_dynamics(counter_time)

            # calculate the amplitude (`new_counter_states[2*i + 1]`)
            current_amplitude = self.counter_states[2*i + 1]
            current_counter = self.counter_states[2*i]
            new_counter_states[2*i + 1] = sp.Piecewise(
                # (new_amplitude, sp.Equality(i, x4_counter_idx)), (1, True)
                (new_amplitude, eq(i, x4_counter_idx)), (0,  eq(current_counter, 0)), (current_amplitude, True)
            )

            partial_bp_effects[i] = self._single_dose_bp_effect(counter_time, current_amplitude)

        # increase the counter index for every nonzero input, but start at 0 again
        # if all counters have been used (achieved by modulo (%))
        x4_counter_idx_new = (x4_counter_idx + sp.Piecewise((1, self.u1 > 0), (0, True))) % self.n_counters

        max3_func = self._define_max3_func()
        assert self.n_counters == 3
        x2_sensitivity_new = max3_func(*partial_sensitivities)

        # IPS()
        x1_bp_effect_new = 100 - 100*sum(partial_bp_effects)

        new_state = [x1_bp_effect_new, x2_sensitivity_new, x3, x4_counter_idx_new, x5_debug] + new_counter_states

        return new_state

    def _single_dose_bp_effect(self, counter_time, amplitude):

        res = self.bp_effect_dynamics_expr.subs(t, counter_time)*amplitude
        return res


    def output(self):
        # sensitivity = sp.Piecewise((self.x2, self.x2 > 1), (1, True))
        # return sensitivity

        return self.x1

    def propofol_bolus_sensitivity_dynamics(self, t):
        """
        :param t:    time since last bolus (sympy expression)
        """

        k = 0.52175
        maxval = 1.26

        t_peak = 1.5

        f1 = - k / (2 + sp.exp(10*t - 5)) + maxval
        f2 = k / (2 + sp.exp(2.5*t - 8)) + 1

        res = sp.Piecewise((f1, t < t_peak), (f2, True))

        return res

    def propofol_bolus_static_values(self, dose: float):
        """
        :param dose:    specific dose in mg/kgBW
        :returns:       effect_of_medication (between 0 and 1)

        """
        k = -0.34655
        effect_of_medication = 1 - sp.exp(k*dose)
        return effect_of_medication



def limit(x, xmin=0, xmax=1, ymin=0, ymax=1):

    dx = xmax - xmin
    dy = ymax - ymin
    m = dy/dx

    new_x_expr = ymin + (x - xmin)*m

    return sp.Piecewise((ymin, x < xmin), (new_x_expr, x < xmax), (ymax, True))


def blocksimulation(k_end, rhs_options=None, iv=None):
    """
    :param k_end:       int; number of steps to simulate
    :param rhs_options: dict; passed to gen_global_rhs
    :param iv:          dict; initial values like {symbol: value, ..}
                        non-specified values are assumed to be 0

    """

    if rhs_options is None:
        rhs_options = {}

    if iv is None:
        iv = {}

    # generate equation system
    rhs_func = gen_global_rhs(**rhs_options)
    initial_state = [0]*len(ds.all_state_vars)

    for symbol, value in iv.items():
        idx = ds.all_state_vars.index(symbol)
        initial_state[idx] = value

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


def gen_global_rhs(use_sp2c=False, use_existing_so=False, sp2c_cleanup=True):
    """
    :param use_sp2c:            bool; whether to use sympy_to_c (instead of lambdify)
    :param use_existing_so:     bool; whether to reuse shared object file if it exists
    :param sp2c_cleanup:        bool; whether to delete auxiliary files for c-code generation
    """

    ds.global_rhs_expr = []
    ds.all_state_vars = []

    rplmts = [(t, k*T)]

    # handle feedback loops
    rplmts.extend(loop_mappings.items())

    # check that none of the target expressions are None
    assert None not in list(zip(*rplmts))[1]

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

    vars = [k, *ds.all_state_vars]
    if use_sp2c:
        assert sp2c_cleanup in (True, False)
        import sympy_to_c as sp2c
        sp2c.core.CLEANUP = sp2c_cleanup

        ds.rhs_func = sp2c.convert_to_c(vars, ds.global_rhs_expr, use_existing_so=use_existing_so)
    else:
        ds.rhs_func = st.expr_to_func(vars, ds.global_rhs_expr, modules="numpy", eltw_vectorize=False)

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
