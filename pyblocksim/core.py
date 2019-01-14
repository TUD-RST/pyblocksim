# -*- coding: utf-8 -*-

"""
Module for block based modeling and simulation of dynamic systems
"""


from __future__ import print_function
from six import string_types  # py2 and 3 compatibility
from collections import OrderedDict

from numpy.lib.index_tricks import r_

import numpy as np
import scipy.integrate as integrate

import sys
import sympy as sp

import inspect
import warnings



def mainprint(*args, **kwargs):
    """
    This function wraps pythons print function such that it is
    only executed if the calling module is the main module.

    This is relevant to the examples which are imported as modules in the
    unittest script where we dont want all the output which is generated,
    by the actual example
    """

    frame_up = inspect.currentframe().f_back
    module_name = frame_up.f_globals['__name__']
    # print("name:", __name__, module_name)

    if module_name == '__main__':
        print(*args, **kwargs)


# The laplace variable
s = sp.Symbol('s')

# The time variable
t = sp.Symbol('t')


def numbered_symbols(prefix='x', function=sp.Symbol,
                                start=0, *args, **assumptions):
    """
    Generate an infinite stream of Symbols consisting of a prefix and
    increasing subscripts.

    This implementation is copied from sympy.numbered_symbols and
    adapted for leading zeros
    """
    while True:
        name = '%s%02i' % (prefix, start)
        yield function(name, *args, **assumptions)
        start += 1


blockoutputs = numbered_symbols('Y')
statevariables = numbered_symbols('SV_')


def _get_assingment_name():
    """
    Supposed to be called in the __init__ of Blocks

    retrieves the Blockname (by inspecting the src code)
    from the left side of the assignments where block is created

    SUM1 =  Blockfunc(...) -> SUM1
    """

    # this is far from elegant and will need to be
    # adapted everytime the class structure changes
    f = inspect.currentframe().f_back.f_back.f_back

    src = inspect.findsource(f)[0][f.f_lineno-1]

    if not "=" in src:
        return 'unnamed'
    else:
        name = src.split("=")[0].strip()
        return name


def degree(expr):
    return sp.Poly(expr,s, domain='EX').degree()


class StateAdmin(object):
    """
    Omniscient object for data bookkeeping
    """
    def __init__(self):
        self.dim = 0
        self.auxdim = 0

        self.dynEqns = []  # list for storing the rhs of state eqns
        self.auxEqns = []  # list for the auxillary eqns
        self.stateVars = []
        self.pseudoStateVars = []  # Symbols to save the output of delay blocks
        self.pseudoStateVarIndices = []  # indices of pseudo state vars
        self.inputs = []

        # use ordered dicts to allow access via name and via index
        self.IBlocks = OrderedDict()
        self.NILBlocks = OrderedDict()
        self.Blockfncs = OrderedDict()
        self.DelayBlocks = OrderedDict()

        self.allBlocks = OrderedDict()
        self.allBlockNames = OrderedDict()
        self.metaBlocks = OrderedDict()
        self.loops = OrderedDict()

        self.blockoutdict = None

    def register_block(self, block):
        """
        Method for central bookkeeping of all blocks
        """
        # TODO: this should be refactored to happen inside the block-Classes

        if isinstance(block, IBlock):
            self._register_IBlock(block)
        elif isinstance(block, NILBlock):
            self._register_NILBlock(block)
        elif isinstance(block, Blockfnc):
            self._register_Blockfnc(block)
        elif isinstance(block, TFBlock):
            self._register_TFBlock(block)
        elif isinstance(block, DelayBlock):
            self._register_DelayBlock(block)
        else:
            raise TypeError()

    def _register_IBlock(self, block):
        self.IBlocks[block.Y] = block
        self.allBlocks[block.Y] = block
        self.allBlockNames[block.name] = block

        block.stateadmin = self
        self._register_new_states(block)
        self._register_dynEqns(block.idcs, block.X, block.expr)

    def _register_NILBlock(self, block):
        self.NILBlocks[block.Y] = block
        self.allBlocks[block.Y] = block
        self.allBlockNames[block.name] = block

    def _register_Blockfnc(self, block):
        self.allBlocks[block.Y] = block
        self.allBlockNames[block.name] = block
        self.Blockfncs[block.Y] = block

    def _register_TFBlock(self, block):
        # prevent the meta block from overwriting the numerator Block
        Y = sp.Symbol(block.Y.name+'meta')
        self.allBlocks[Y] = block
        self.allBlockNames[block.name] = block
        self.metaBlocks[block.Y] = block

    def _register_DelayBlock(self, block):
        self.DelayBlocks[block.Y] = block
        self.allBlocks[block.Y] = block
        self.allBlockNames[block.name] = block

        block.stateadmin = self
        self._register_new_states(block)
        self.pseudoStateVars.extend(block.stateVars)
        self.pseudoStateVarIndices.extend(block.idcs)

        # TODO: find a more elegant way
        # the corresponding equation for the pseudo state
        # is not important, we just need some expression
        self.dynEqns.append(sp.sympify(0))

    def _register_new_states(self, block):
        """
        :param block:

        Create and distribute symbols for the state variables for the passed block.
        This method is also used for pseudo states (delay blocks)
        """
        # let the block know which indices belong to it
        block.idcs = list(range(self.dim, self.dim+block.order))
        self.dim += block.order

        block.stateVars = [next(statevariables) for i in block.idcs]
        self.stateVars += block.stateVars

    def _register_new_auxSignal(self):
        self.auxdim += 1
        return self.auxdim-1

    def _register_dynEqns(self, idcs, insig, expr):
        """

        :param idcs:    indices w.r.t the overall state vector
        :param insig:   input signal
        :param expr:    denominator expression (like s**2 + 3s - 2)
        :return:
        """
        coeffs = expr2coeffs(expr)
        order = degree(expr)

        assert not order == 0

        self.dynEqns.extend([None]*len(idcs))

        # the following requires that the respective state vars have already been created and
        # are stored in self.statevars

        # handle integrator chains like xdot1 = x2
        tmpVars = []
        for i in idcs[0:-1]:
            tmpVars.append(self.stateVars[i+1])
            self.dynEqns[i] = tmpVars[-1]

        # create the last line of the controller canonical form
        tmpVars.insert(0, self.stateVars[idcs[0]])
        inputeqn = sum([-c*v for c, v in zip(coeffs[:-1], tmpVars) ]) + insig
        self.dynEqns[idcs[-1]] = inputeqn

    def register_inputs(self, namestr):

        res = sp.symbols(namestr) # returns one symbol or a tuple of symbols

        if isinstance(res, sp.Symbol):
            res = [res]
        res = list(res)

        self.inputs += res
        return res

    def register_loop(self, output, input_to_be_replaced):
        # TODO: loop with state-Variable not yet tested
        assert output in self.allBlocks or output in self.stateVars
        assert input_to_be_replaced in self.inputs

        self.loops.update({input_to_be_replaced : output})

    def get_nil_eq(self, nilblock):
        """
        computes the eqn for a Non Integrating Linear block
        (i. e. collects the derivatives of the input and
        combines them linearly (as the polynomial says)
        """
        assert isinstance(nilblock, NILBlock)

        if nilblock.X in self.inputs and len(nilblock.coeffs) > 1:
            raise NotImplementedError("Input derivative")

        if not nilblock.X in self.IBlocks:
            raise ValueError("Invalid input signal of " + str(nilblock))

        prevblock = self.IBlocks[nilblock.X]

        # n = len(nilblock.coeffs)-1

        # coeffs are sorted like this: [a_0, ..., a_n]
        tmplist = [c*prevblock.requestDeriv(i)
                                for i, c in enumerate(nilblock.coeffs)]

        formula = sum(tmplist)

        return formula

    def resolve_blockfnc(self, bf, subsdict):
        """
        go down the algebraic chain to substitue for all other algebraic
        iterim results

        subsdict is supposed to contain the mapping from the block outputs
        of IBlocks and NILBlocks to the state variables
        """

        # get all inputs of that block which cannot be represented as statevars
        # easily
        fnc_symbs = [sy for sy in bf.symbs if sy in self.Blockfncs]

        results = {}
        # this could be improved by caching
        for fs in fnc_symbs:
            fnc_block = self.Blockfncs[fs]
            res = self.resolve_blockfnc(fnc_block, subsdict)
            results.update({fs: res})

        new_subs_dict = subsdict.copy()
        new_subs_dict.update(results)
        return bf.expr.subs(new_subs_dict)

    def setup_DelayBlocks(self, dt):
        # idea: setup the delay blocks before the simulation starts
        # at that time dt is known and so we can calculate the
        # buffer length
        for key, block in self.DelayBlocks.items():
            block._setup(dt)
# End of class StateAdmin


def expr2coeffs(expr, lead_test=True):
    """ returns a list of the coeffs (highest last)
    """

    # coeffs = sp.Poly(expr, s).coeffs # -> only non-zero coeffs
    p = sp.Poly(expr, s, domain="EX")

    # !! this should be done with .all_coeffs() if its available
    # then we probably must reverse the array because highest coeff will be 1st
    # -> coeffs = np.array(map(float, coeffs))[::-1]

    c_dict = p.as_dict()
    coeffs = [c_dict.get((i,), 0) for i in range(p.degree()+1)]
    # convert to np array
    coeffs = np.array(list(map(float, coeffs)))

    # check if leading coeff == 1
    if lead_test:
        assert coeffs[-1] == 1
    return coeffs


class AbstractBlock(object):

    def __init__(self, name):
        if name is None:
            name_candidate = _get_assingment_name()
        else:
            if not isinstance(name, string_types):
                raise TypeError("invalid block name")
            name_candidate = name
            if name_candidate in theStateAdmin.allBlockNames:
                msg = "Warning: explicitly given block name '{0}' already exists. "\
                      "It will be renamed."
                warnings.warn(msg.format(name))

        tmp_name = name_candidate
        i = 1
        while tmp_name in theStateAdmin.allBlockNames:
            tmp_name = name_candidate + "_" + str(i)

        self.name = tmp_name

    def __repr__(self):
        return type(self).__name__+':'+self.name

    # TODO: This method has not yet fully been testet
    def connect_new_input(self, insig_new):
        """
        Allows to reconfigure the network, e.g. to study the
        consequences of different topology. Use with care.
        """
        assert hasattr(self, 'X')
        assert self.X is not None

        self.X = insig_new

        if hasattr(self, 'denomBlock'):
            self.denomBlock.connect_new_input(insig_new)


class IBlock(AbstractBlock):
    """
    Integrating block
    """
    def __init__(self, expr, insig, name = None):
        """
        expr ... the denominator expr. (a sympy polynomial)
        """
        AbstractBlock.__init__(self, name)
        self.X = insig
        self.Y = next(blockoutputs)

        self.coeffs = expr2coeffs(expr)
        self.order = degree(expr)
        self.expr = expr

        # placeholder:
        self.idcs = None
        self.stateVars = None
        self.stateadmin = None  # ugly and awkward

        theStateAdmin.register_block(self)

    def requestDeriv(self, n):
        assert int(n) == n
        assert n >= 0

        if n < self.order:
            return self.stateVars[n]

        elif n == self.order:
            return self.stateadmin.dynEqns[self.idcs[-1]]

        else:
            raise NotImplementedError("derivative propagation not yet " \
                                       "supported")


# TODO: should be renamed to RTFBlock (due to rational TF)
class TFBlock(AbstractBlock):
    """
    Metablock for representing rational transfer functions
    will be decomposed to numerator (NILBlock) and denom. (IBlock)
    """

    def __init__(self, expr, insig, name=None):

        AbstractBlock.__init__(self, name)

        num, denom = expr.as_numer_denom()

        highest_denom_coeff = expr2coeffs(denom, lead_test=False)[-1]

        num /= highest_denom_coeff
        denom /= highest_denom_coeff

        self.denom_degree = degree(denom)
        self.relative_degree = degree(denom) - degree(num)
        self.num = num

        if self.denom_degree == 0:
            assert degree(num) == 0

            self.denomBlock = None
            # self.numBlock = NILBlock(num, insig, name+'_num')
            self.numBlock = Blockfnc(num * insig)
        else:
            self.denomBlock = IBlock(denom, insig, self.name+'_denom')
            self.numBlock = NILBlock(num, self.denomBlock.Y, self.name+'_num')

        self.X = insig
        self.Y = self.numBlock.Y
        self.output_deriv_cache = {}

        theStateAdmin.register_block(self)

    def get_output_deriv(self, order):
        """
        Creates an additional numerator (-> NILBlock) for the TF which serves to
        calculate derivatives of the actual output

        :param order: derivative order; must be smaller than relative degree

        :return: output variable of the new NILBlock
        """

        assert int(order) == order
        if order > self.relative_degree:
            msg = "Output derivative order must not be greater " \
                  "than relative degree."
            raise ValueError(msg)

        if order == 0:
            return self.Y

        if self.output_deriv_cache.get(order) is not None:
            return self.output_deriv_cache.get(order)

        # Now we have to create a new NILBlock for the new numerator

        block_name = self.name+'_num_d' + str(order)
        new_numerator = NILBlock(self.num*s**order, self.denomBlock.Y, block_name)
        self.output_deriv_cache[order] = new_numerator
        return new_numerator.Y


class NILBlock(AbstractBlock):
    """
    Non-Integrating Linear Block (we call it NIL-Block)
    """
    def __init__(self, expr, insig, name = None):
        """
        expr ... the denominator expr. (a sympy polynomial)
        """
        self.X = insig
        self.Y = next(blockoutputs)

        AbstractBlock.__init__(self, name)

        self.coeffs = expr2coeffs(expr, False)
        self.order = sp.Poly(expr, s, domain='EX').degree()

        theStateAdmin.register_block(self)


class Blockfnc(AbstractBlock):
    """
    used to realize linear and nonlinear static functions
    """

    def __init__(self, expr, name=None):
        atoms = expr.atoms()

        assert not s in atoms, "This is just for static functions"

        symbs = [a for a in atoms if not a.is_number]
        self.symbs = symbs

        AbstractBlock.__init__(self, name)

        tmpList = list(theStateAdmin.allBlocks.keys()) + theStateAdmin.inputs + \
                    theStateAdmin.stateVars
        assert all([sy in tmpList for sy in symbs])

        self.Y = next(blockoutputs)

        theStateAdmin.register_block(self)

        self.expr = expr


class DelayBlock(AbstractBlock):

    def __init__(self, T, insig, ivalue=None, name=None):
        """
        :param T:       delaytime
        :param insig:   input signal of the block
        :param ivalue   initial value (array or function);
                        not yet supported
        :param name:
        """
        AbstractBlock.__init__(self, name)
        self.T = T

        self.X = insig
        self.Y = next(blockoutputs)

        self.stateadmin = None

        # There will be one pseudo state to store the output of this block
        self.order = 1
        # this will be lenght-1-sequences after registration
        self.idcs = None
        self.stateVars = None

        self.input_value_fnc = None
        # this will be the ringbuffer
        self.rb = None

        if ivalue is not None:
            raise NotImplementedError("Initial value not yet supported")

        theStateAdmin.register_block(self)

    def _setup(self, dt):
        """
        should be called beor
        :param dt:
        :return:
        """
        length = int(self.T / dt)
        self.rb = RingBuffer(length)

    def read(self):
        return self.rb.read()

    def write_input_and_step(self, input_value):
        return self.rb.write_and_step(input_value)


class RingBuffer(object):
    """
    data structure for saving the internal state of a delay block
    """

    def __init__(self, length):
        self._storrage = np.zeros(length)
        self._length = length
        self._idx = 0
        self._flag_read = False

    def read(self):
        assert not self._flag_read
        self._flag_read = True
        return self._storrage[self._idx]

    def write_and_step(self, value):
        assert np.allclose(np.float64(value), value)

        # ensure we already read the value
        assert self._flag_read

        self._storrage[self._idx] = value
        self._idx = (self._idx + 1) % self._length
        self._flag_read = False


def exceptionwrapper(fnc):
    """
    prevent the integration algorithm to get stuck if
    a exception occurs in rhs
    """

    def newfnc(*args, **kwargs):
        try:
            return fnc(*args, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)

    return newfnc


# TODO this function is too long
def gen_rhs(stateadmin):
    """
    resolves dependencies in the eqns
    and creates a function (called rhs) which can be passed to the
    integration algorithm
    """

    subsdict = {}

    # handle integrating blocks (Yii <- SV_jj)
    for y, bl in stateadmin.IBlocks.items():
        subsdict.update({y: stateadmin.stateVars[bl.idcs[0]]})

    # handle NIL Blocks
    for y, bl in stateadmin.NILBlocks.items():
        eqn_rhs = stateadmin.get_nil_eq(bl)  # can still contain Yii -vars
        subsdict.update({y: eqn_rhs})

    # handle Blockfncs
    fncs = {}
    for y, bl in stateadmin.Blockfncs.items():
        new_y = stateadmin.resolve_blockfnc(bl, subsdict)
        fncs[y] = new_y

    subsdict.update(fncs)

    # handle delay blocks:
    stateadmin.delayblockoutputs = []

    for k, v in stateadmin.DelayBlocks.items():
        stateadmin.delayblockoutputs.append(k)
        # associate the pseudo state variable
        subsdict[k] = psv = v.stateVars[0]
        assert psv in stateadmin.pseudoStateVars

    # now eliminate the Yii vars in the expressions:
    Yii = set(subsdict.keys())
    finished_expr = {}

    L0 = len(subsdict)
    for y, expr in list(subsdict.items()):
        if expr.atoms().intersection(Yii) == set():
            finished_expr[y] = expr
            subsdict.pop(y)

    while True:
        L = len(subsdict)
        for y, expr in list(subsdict.items()):
            expr = expr.subs(finished_expr)
            if expr.atoms().intersection(Yii) == set():
                finished_expr[y] = expr
                subsdict.pop(y)
            else:
                # not ready yet, but maybe next round
                subsdict[y] = expr # update the subsdict

        assert len(subsdict) < L or L == 0
        if subsdict == {}:
            break

    assert len(finished_expr) == L0

    subsdict = finished_expr

    # close the loops
    loops = {}
    for u, y in stateadmin.loops.items():
        new_y = y.subs(subsdict)
        loops[u] = new_y

    # check for algebraic loops (too strict!! but OK for now)
    # we don't allow a Blockfnc output to be fed back
    # so we ensure that there is at least one integrator in the loop
    L = set(loops).intersection(stateadmin.Blockfncs)
    if not L == set():
        raise ValueError('Algebraic loop found (maybe): '+str(L))

    for u, expr in loops.items():
        if expr.atoms().intersection(stateadmin.allBlocks) != set():
            raise ValueError('unsubstituted expr found: %s=%s' %(u, expr))

    for y, expr in subsdict.items():
        subsdict[y] = expr.subs(loops)

    # we do not need them anymore
    # remove them to avoid confusion
    # but this cause problems if the gen_rhs is called more than once
#    for u in loops:
#        theStateAdmin.inputs.remove(u)

    # save the relations for later use
    stateadmin.blockoutdict = subsdict

    args = stateadmin.stateVars + stateadmin.inputs +\
           stateadmin.delayblockoutputs

    state_rhs_fncs = []
    stateadmin.final_equations = []
    stateadmin.args = args

    # the inputs and parameters are taken from global scope (w.r.t. rhs)
    for eq in stateadmin.dynEqns:
        eq = eq.subs(subsdict)
        stateadmin.final_equations.append(eq)
        fnc = sp.lambdify(args, eq)
        state_rhs_fncs.append(fnc)


    def rhs(z, t, *addargs):
        fncargs = list(z)+list(addargs)
        dz = [fnc(*fncargs) for fnc in state_rhs_fncs]
        return dz

    return exceptionwrapper(rhs)


def compute_block_ouptputs(simresults):

    args = theStateAdmin.stateVars + theStateAdmin.inputs
    assert simresults.shape[1] == len(args)

    blocks = {}

    blockout_vars = set(theStateAdmin.allBlocks)

    for v in list(theStateAdmin.blockoutdict.values()):
        # on the right hand side no Yii should be found
        to_be_empty = v.atoms().intersection(blockout_vars)

        if not to_be_empty == set():
            raise ValueError("This set should be empty: ", to_be_empty)

    for bl in list(theStateAdmin.allBlocks.values()):

        y = bl.Y
        # the fnc for calculationg the blockoutput from states
        fnc = sp.lambdify(args, y.subs(theStateAdmin.blockoutdict))

        numargs = list(simresults.T)
        # -> list of N 1d-arrays, N=number(statevars)+number(inputs)

        # for each timestep: evaluate the block specific func with the args
        blocks[bl] = tmp = np.array([fnc(*na) for na in zip(*numargs)])
        try:
            float(tmp[0])
        except TypeError as e:
            tp = type(tmp[0])
            msg = "Invalid type ({}) while computing output of block: {}.".format(tp, bl.name)
            e.args = (msg, )
            raise

    return blocks


def blocksimulation(tend, inputs=None, xx0=None, dt=5e-3):
    """
    xx0       state for t = 0
    inputs    a dict (like {u1 : calc_u1, u2 : calc_u2} where calc_u1, ...
              are callables
              (in the case of just one specified input (u1, calc_u1) is allowed
    """

    if xx0 is None:
        xx0 = [0]*theStateAdmin.dim

    assert len(xx0) == theStateAdmin.dim

    assert float(tend) == tend and tend > 0

    if isinstance(inputs, dict):
        pass
    elif inputs is None:
        inputs = {}

    # shortcut for 2-tuple of the form (u1, u1fnc)
    elif hasattr(inputs, '__len__') and len(inputs) == 2 and\
                                    not hasattr(inputs[0], '__len__'):
        inputs = dict([inputs])
    else:
        msg = "invalid type for input: " + str(input)
        raise TypeError(msg)

    allinputs = {}
    for i in theStateAdmin.inputs:
        allinputs[i] = lambda t: 0  # dummy functions

    allinputs.update(inputs)
    inputfncs = [allinputs[i] for i in theStateAdmin.inputs]

    # generating the model from the blocks
    theStateAdmin.rhs = rhs = gen_rhs(theStateAdmin)

    t = 0
    x_vect = xx0

    # input vector
    u_vect = np.array([fnc(0) for fnc in inputfncs])

    # vector of delay block outputs:
    theStateAdmin.setup_DelayBlocks(dt)
    delayblocks = theStateAdmin.DelayBlocks.values()

    # TODO: this should be a separat function
    # the input of a deayblock is a) an system input
    # or b) an other blockoutput

    # will be a list of tuples:
    potential_rhs_exprns = list(theStateAdmin.blockoutdict.items())
    # add trivial equations for the input (like u1 = u1)
    tuple_list = zip(theStateAdmin.inputs, theStateAdmin.inputs)
    potential_rhs_exprns.extend(tuple_list)

    args = theStateAdmin.stateVars + theStateAdmin.inputs

    for block in delayblocks:
        expr = block.X.subs(potential_rhs_exprns)
        fnc = sp.lambdify(args, expr)
        block.input_fnc = fnc

        # save the expression of the lambdified function for debug purposes
        block.input_fnc.expr = expr

    d_vect = np.array([block.read() for block in delayblocks])

    assert len(theStateAdmin.pseudoStateVarIndices) == len(d_vect)

    # create an empty array to which the results will by added
    arr_length = len(x_vect) + len(u_vect)
    stateresults = np.array([]).reshape(0, arr_length)

    tvect = np.array([])

    while True:
        # save the current values
        stateresults = np.vstack((stateresults, r_[x_vect, u_vect]))
        tvect = np.hstack((tvect, t))

        if t >= tend:
            break

        # calculate the next values
        addargs = tuple(u_vect) + tuple(d_vect)
        x_vect = integrate.odeint(rhs, x_vect, r_[t, t+dt], addargs)
        x_vect = x_vect[-1, :]

        # save the value of delay block in the corresponding pseudo state
        for idx, d_value in zip(theStateAdmin.pseudoStateVarIndices, d_vect):
            x_vect[idx] = d_value

        t += dt
        u_vect = [fnc(t) for fnc in inputfncs]

        # handle delay blocks

        for i, block in enumerate(delayblocks):
            value = block.input_fnc(*stateresults[-1, :])
            block.write_input_and_step(value)
            d_vect[i] = block.read()

    tvect = tvect

    return tvect, stateresults


def get_linear_ct_model(stateadmin, system_output):
    """
    Return Matrices A, B, C, D


    :param stateadmin:
    :return:
    """

    gen_rhs(stateadmin)

    sys_eqns = sp.Matrix(stateadmin.final_equations)
    xx = sp.Matrix(stateadmin.stateVars)
    uu = sp.Matrix([u for u  in stateadmin.inputs if u not in theStateAdmin.loops])

    # symbolic matrices
    As = sys_eqns.jacobian(xx)
    Bs = sys_eqns.jacobian(uu)

    system_output2 = system_output.subs(theStateAdmin.blockoutdict)

    Cs = system_output2.jacobian(xx)
    Ds = system_output2.jacobian(uu)

    # test that we have a linear system (no symbols occur in the matrices -> assert empty set)
    assert not As.atoms(sp.Symbol)
    assert not Bs.atoms(sp.Symbol)
    assert not Cs.atoms(sp.Symbol)
    assert not Ds.atoms(sp.Symbol)

    # create numeric arrays
    A, B, C, D = (np.array(np.array(M), dtype=np.float) for M in (As, Bs, Cs, Ds))

    return A, B, C, D



def stepfnc(tup, amp1=1, tdown = np.inf, amp0=0):
    """
    returns a callable of 1 arg which is a step function
    """
    assert float(tup) == tup
    assert float(tdown) == tdown
    assert float(amp1) == amp1
    assert float(amp0) == amp0

    assert tdown > tup

    def fnc(t):
        if t < tup :
            u = amp0
        elif t < tdown :
            u = amp1
        else:
            u = amp0
        return u

    return fnc


class Trajectory(object):
    def __init__(self, expr, smoothness_degree):
        self.expr = expr
        self.sd = smoothness_degree
        self.derivatives = []
        self._initialize()

    def _initialize(self):
        assert isinstance(self.expr, sp.Expr)
        assert self.expr.atoms(sp.Symbol) == set([t])
        assert int(self.sd) == self.sd

        for i in range(self.sd+1):
            self.derivatives.append( self.expr.diff(t, i) )

    def _make_fnc(self, expr, var=None):
        if var is None:
            var = t
        return sp.lambdify( (var,), expr, modules='numpy' )

    def get_trajectory(self, diff_idx=0):
        assert diff_idx <= self.sd  # smoothness_degree
        fnc = self._make_fnc(self.derivatives[diff_idx])
        return fnc

    def combined_trajectories(self, expr):
        """ expects expr to be a polynomial in s which determines
        how to linearily combine the trajectory
        and its derivatives to a new function of time

        Example : expr = s**2 -3*s + 5 means

        return a fuction consisting of y_ddot - 3*y_dot + 5*y

        """

        coeffs = expr2coeffs(expr, lead_test=False)

        res = 0
        for i, c in enumerate(coeffs):
            res += c*self.derivatives[i]

        return self._make_fnc(res)


def main():
    print("This module is not meant to be executed as main module.")
    print("See examples")


def restart():
    """
    Forget about all blocks and states. This is useful for unittests
    """
    theStateAdmin.__init__()

# global variables
theStateAdmin = StateAdmin()
loop = theStateAdmin.register_loop
inputs = theStateAdmin.register_inputs


if __name__ == '__main__':
    main()
