"""
This module contains a toolbox to construct, simulate and postprocess a graph of time discrete blocks
"""

from typing import List
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function

import symbtools as st

from ipydex import IPS

# abbreviation for the equality operator (needed in some Piecewise definitions)
eq = sp.Equality

# discrete time

k = sp.Symbol("k")

# this will be replaced by k*T in the final equations
t = sp.Symbol("t")

# discrete step time
T = 0.1

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


# TODO: write unittest for this mechanism
def get_loop_symbol():
    ls = next(loop_symbol_iterator)
    loop_mappings[ls] = None
    return ls


def set_loop_symbol(ls, expr):
    ensure_scalar(expr)
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
        self.numbered_state_symbols = sp.numbered_symbols("x", start=1)
        self.numbered_input_symbols = sp.numbered_symbols("u", start=1)

        self.global_rhs_expr = None
        self.global_input_expr_list = None
        self.all_state_vars = None
        self.all_input_vars = None
        self.rhs_func = None
        self.input_func = None
        self.state_history = None
        self.input_history = None

    def get_state_vars(self, n) -> List[sp.Symbol]:
        res = [next(self.numbered_state_symbols) for i in range(n)]
        return res

    def get_input_vars(self, m) -> List[sp.Symbol]:
        res = [next(self.numbered_input_symbols) for i in range(m)]
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

        self.set_inputs(input1, **kwargs)

        self.n_inputs = len(self.input_expr_list)
        self.input_vars = ds.get_input_vars(self.n_inputs)

        # assign input vars to attributes of the block (like `self.u1``)
        for i, var in enumerate(self.input_vars, start=1):
            setattr(self, f"u{i}", var)

        # params provided (hardcoded) by the class

        self.params = self._class_specific_params()

        # params provided by the constructor (can overwrite class parameters)
        if params is not None:
            assert isinstance(params, dict)
            self.params.update(params)

        # save them as attributes
        self.__dict__.update(self.params)

        # make each state variable available as `self.x1`, `self.x2`, ...
        for i, x_i in enumerate(self.state_vars, start=1):
            setattr(self, f"x{i}", x_i)

        if name is None:
            name = f"{type(self).__name__}__{self.instance_counter}"
        self.name = name

        # store implemented functions (e.g. for counter handling)
        self._implemented_functions = {}

        ds.register_block(self)

    def _class_specific_params(self):
        """
        enables to specify parameters in subclasses
        """
        return {}


    def set_inputs(self, input1, **kwargs):

        if input1 is None:
            input1 = sp.sympify(0)
        rest_list = self._get_input_exprs_from_kwargs(kwargs)

        # this might be called before the attribute is created
        tmp_input_expr_list = getattr(self, "input_expr_list", None)
        if tmp_input_expr_list is not None:
            assert len(self.input_expr_list) == len(rest_list) + 1
            assert len(self.input_vars) == len(rest_list) + 1

        self.input_expr_list = [input1, *rest_list]
        self._check_inputs()

    def _check_inputs(self):
        for expr in self.input_expr_list:
            ensure_scalar(expr)

    def _get_input_exprs_from_kwargs(self, kwargs: dict, set_attrs=False):

        input_exprs = []
        i = 1
        for key, value in kwargs.items():
            i += 1
            # assume key like input3
            assert key.startswith("input")
            assert i == int(key.replace("input", ""))
            if set_attrs:
                # setattr(self, f"u{i}", value)
                raise NotImplementedError("obsolete")
            input_exprs.append(value)
        return input_exprs


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


def tmp_eq_fnc(x4, i, res):
    if x4 == i:
        return res
    else:
        return 0

# TODO: obsolete?
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


def limit(x, xmin=0, xmax=1, ymin=0, ymax=1):

    dx = xmax - xmin
    dy = ymax - ymin
    m = dy/dx

    new_x_expr = ymin + (x - xmin)*m

    return sp.Piecewise((ymin, x < xmin), (new_x_expr, x < xmax), (ymax, True))


def blocksimulation(k_end, rhs_options=None, iv=None, calc_outputs=True):
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

    # generate equation system (if necessary)
    if ds.rhs_func is None:
        rhs_func = gen_global_rhs(**rhs_options)
    else:
        rhs_func = ds.rhs_func

    # create initial state
    initial_state = [0]*len(ds.all_state_vars)

    for symbol, value in iv.items():
        idx = ds.all_state_vars.index(symbol)
        initial_state[idx] = value

    # solve equation system
    current_state = initial_state
    ds.state_history = []
    ds.input_history = []

    kk_num = np.arange(k_end)
    for k_num in kk_num:
        ds.state_history.append(current_state)
        new_input = ds.input_func(k_num, *current_state)
        ds.input_history.append(new_input)
        current_state = rhs_func(k_num, *current_state, *new_input)

    # postprocessing
    ds.state_history = np.array(ds.state_history)
    ds.input_history = np.array(ds.input_history)

    if calc_outputs:
        block_outputs = compute_block_outputs(kk_num, ds.state_history)
    else:
        block_outputs = None

    return kk_num, ds.state_history, block_outputs


def gen_global_rhs(use_sp2c=False, use_existing_so=False, sp2c_cleanup=True):
    """
    :param use_sp2c:            bool; whether to use sympy_to_c (instead of lambdify)
    :param use_existing_so:     bool; whether to reuse shared object file if it exists
    :param sp2c_cleanup:        bool; whether to delete auxiliary files for c-code generation
    """

    ds.global_rhs_expr = []
    ds.all_state_vars = []
    ds.all_input_vars = []

    rplmts = [(t, k*T)]

    # handle feedback loops
    rplmts.extend(loop_mappings.items())

    # check that none of the target expressions are None
    assert None not in list(zip(*rplmts))[1]

    for block_name, state_vars in ds.state_var_mapping.items():
        block_instance: TDBlock = ds.block_instances[block_name]

        rhs_expr = list(block_instance.rhs(k, state_vars))

        ds.all_state_vars.extend(state_vars)
        ds.all_input_vars.extend(block_instance.input_vars)

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

    vars = [k, *ds.all_state_vars, *ds.all_input_vars]
    if use_sp2c:
        assert sp2c_cleanup in (True, False)
        import sympy_to_c as sp2c
        sp2c.core.CLEANUP = sp2c_cleanup

        ds.rhs_func = sp2c.convert_to_c(vars, ds.global_rhs_expr, use_existing_so=use_existing_so)
    else:
        ds.rhs_func = st.expr_to_func(vars, ds.global_rhs_expr, modules="numpy", eltw_vectorize=False)

    generate_input_func()

    return ds.rhs_func


def generate_input_func():

    rplmts = [(t, k*T)]

    # handle feedback loops
    rplmts.extend(loop_mappings.items())

    ds.global_input_expr_list = []
    for block_name, _ in ds.state_var_mapping.items():
        block_instance: TDBlock = ds.block_instances[block_name]

        for expr in block_instance.input_expr_list:
            ds.global_input_expr_list.append(sp.sympify(expr).subs(rplmts))

    # some inputs might depend on state components (e.g. for concatenated blocks)
    vars = [k, *ds.all_state_vars]

    ds.input_func = st.expr_to_func(vars, ds.global_input_expr_list, modules="numpy", eltw_vectorize=False)


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


def ensure_scalar(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.MatrixBase):
        # isinstance(expr, sp.Basic) might be also true -> ignore it here
        msg = "Unexpectedly got matrix where scalar was expected."
    elif isinstance(expr, sp.Basic):
        # this is now OK
        return
    else:
        msg = f"Unexpectedly got {type(expr)} where scalar was expected."
    raise TypeError(msg)