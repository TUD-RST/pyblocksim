# -*- coding: utf-8 -*-

"""
Module for block based modeling and simulation of dynamic systems
"""


from __future__ import division

from numpy.lib.index_tricks import r_

import numpy as np
import scipy as sc
import scipy.integrate as integrate

import pylab as pl


import sys
import sympy as sp



import inspect


__version__ = '0.1'



# The laplace variable
s = sp.Symbol('s')

# The time variable
t = sp.Symbol('t')


def numbered_symbols(prefix='x', function=sp.Symbol,
                                start=0, *args, **assumptions):
    """
    copied from sympy.numbered_symbols and adapted for leading zeros
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
        
        self.dynEqns = [] # list for storing the rhs of state eqns
        self.auxEqns = [] # list for the auxillary eqns
        self.stateVars = []
        self.inputs = []
        
        self.IBlocks = {}
        self.NILBlocks = {}
        self.Blockfncs = {}
        
        self.allBlocks = {}
        self.metaBlocks = {}
        self.loops = {}
        
        self.blockoutdict = None
        
    def register_block(self, block):
        
        if isinstance(block, IBlock):
            self._register_IBlock(block)
        elif isinstance(block, NILBlock):
            self._register_NILBlock(block)
        elif isinstance(block, Blockfnc):
            self._register_Blockfnc(block)
        elif isinstance(block, TFBlock):
            self._register_TFBlock(block)
        else:
            raise TypeError
    
    def _register_IBlock(self, block):
        self.IBlocks[block.Y] = block
        self.allBlocks[block.Y] = block
        
        block.stateadmin = self
        self._register_new_states(block)
        self._register_dynEqns(block.idcs, block.X, block.expr)  

    def _register_NILBlock(self, block):
        self.NILBlocks[block.Y] = block
        self.allBlocks[block.Y] = block
                
    def _register_Blockfnc(self, block):
        self.allBlocks[block.Y] = block
        self.Blockfncs[block.Y] = block
    
    def _register_TFBlock(self, block):
        # prevent the meta block from overwriting the numerator Block
        Y = sp.Symbol(block.Y.name+'meta')
        self.allBlocks[Y] = block
        self.metaBlocks[block.Y] = block
    
    def _register_new_states(self, block):        
        # let the block know which indices belong to it
        block.idcs = range(self.dim, self.dim+block.order)
        self.dim += block.order
        
        block.stateVars = [statevariables.next() for i in block.idcs]
        self.stateVars += block.stateVars
        
    def _register_new_auxSignal(self):
        self.auxdim += 1
        return self.auxdim-1
        
    def _register_dynEqns(self, idcs, insig, expr):
        coeffs = expr2coeffs(expr)
        order = degree(expr)
        
        assert not order == 0
        
        self.dynEqns.extend([None]*len(idcs))
        
        tmpVars = []
        for i in idcs[0:-1]:
            tmpVars.append(self.stateVars[i+1])
            self.dynEqns[i] = tmpVars[-1]

        tmpVars.insert(0, self.stateVars[idcs[0]])
        inputeqn = sum([-c*v for c,v in zip(coeffs[:-1], tmpVars) ]) + insig
        self.dynEqns[idcs[-1]] = inputeqn
        
    def register_inputs(self, namestr):
        
        res = sp.symbols(namestr) # returns one symbol or a tuple of symbols
        
        if isinstance(res, sp.Symbol):
            res = [res]
        res = list(res)
        
        self.inputs += res
        return res
    
    def register_loop(self, output, input_to_be_replaced):
        assert output in self.allBlocks
        assert input_to_be_replaced in self.inputs
        
        self.loops.update({input_to_be_replaced : output})
        
    def get_nil_eq(self, nilblock):
        """
        computes the eqn for a Non Integrating Linear block
        (i. e. collects the derivatives of the input and 
        combines them linearly (as the polynomial says)
        """
        assert isinstance(nilblock, NILBlock)
        
        if nilblock.X in self.inputs and len(nilblock.coeffs)>1:
            raise NotImplementedError, "Input derivative"
        
        if not nilblock.X in self.IBlocks:
            raise ValueError, "Invalid input signal of "+ str(nilblock)
        
        prevblock = self.IBlocks[nilblock.X]
        
        #n = len(nilblock.coeffs)-1
        
        # coeffs are sorted like this: [a_0, ..., a_n]
        tmplist = [c*prevblock.requestDeriv(i)
                                for i,c in enumerate(nilblock.coeffs)]
        
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
        fnc_symbs = filter(lambda sy: sy in self.Blockfncs, bf.symbs)
        
        results = {}
        # this could be improved by caching
        for fs in fnc_symbs:
            fnc_block = self.Blockfncs[fs]
            res = self.resolve_blockfnc(fnc_block, subsdict)
            results.update({fs: res})
        
        new_subs_dict = subsdict.copy()
        new_subs_dict.update(results)    
        return bf.expr.subs(new_subs_dict)

        
def expr2coeffs(expr, lead_test = True):
    """ returns a list of the coeffs (highest last)
    """
    
    #coeffs = sp.Poly(expr, s).coeffs # -> only non-zero coeffs
    p = sp.Poly(expr, s, domain="EX")
    
    # !! this should be done with .all_coeffs() if its available
    # then we probably must reverse the array because highest coeff will be 1st 
    # -> coeffs = np.array(map(float, coeffs))[::-1]


    c_dict = p.as_dict()
    coeffs = [c_dict.get((i,), 0) for i in range(p.degree()+1)]
    #convert to np array
    coeffs = np.array(map(float, coeffs))
    
    # check if leading coeff == 1
    if lead_test:
        assert coeffs[-1] == 1    
    return coeffs
        

class AbstractBlock(object):
    
    def __init__(self, name):
        if name == None:
            self.name = _get_assingment_name()
        else:
            self.name = name
    
    def __repr__(self):
        return type(self).__name__+':'+self.name
    

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
        self.Y = blockoutputs.next() 
        
        self.coeffs = expr2coeffs(expr)
        self.order = degree(expr)
        self.expr = expr
        
        # placeholder:
        self.idcs = None 
        self.stateVars = None
        self.stateadmin = None # ugly and awkward 
        
        theStateAdmin.register_block(self)
        
    def requestDeriv(self, n):
        assert int(n) == n
        assert n >= 0
        
        if n < self.order:
            return self.stateVars[n]
        
        elif n == self.order:
            return self.stateadmin.dynEqns[self.idcs[-1]] 
        
        else:
            raise NotImplementedError, "derivative propagation not yet "\
                                       "supported"
        
class TFBlock(AbstractBlock):
    """
    Metablock for representing polynomial transfer functions
    will be decomposed to numerator (NILBlock) ande denom. (IBlock)
    """
    
    def __init__(self, expr, insig, name = None):
        
        AbstractBlock.__init__(self, name)
        
        num, denom = expr.as_numer_denom()
        
        highest_coeff = expr2coeffs(denom, lead_test = False)[-1]
        
        num/= highest_coeff
        denom/= highest_coeff
        
        if degree(denom) == 0:
            assert degree(num) == 0
            
            self.denomBlock = None # ?? maybe a dummy block
            #self.numBlock = NILBlock(num, insig, name+'_num')
            self.numBlock = Blockfnc(num * insig)
        else:
            self.denomBlock = IBlock(denom, insig, self.name+'_denom')
            self.numBlock = NILBlock(num, self.denomBlock.Y, self.name+'_num')
        
        self.X = insig
        self.Y = self.numBlock.Y
        
        theStateAdmin.register_block(self)       
        

class NILBlock(AbstractBlock):
    """
    Non-Integrating Linear Block (we call it NIL-Block)
    """
    def __init__(self, expr, insig, name = None):
        """
        expr ... the denominator expr. (a sympy polynomial)
        """
        self.X = insig
        self.Y = blockoutputs.next() 
        
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

        symbs = filter(lambda a: not a.is_number, atoms)
        self.symbs = symbs
        
        AbstractBlock.__init__(self, name)
        
        tmpList = theStateAdmin.allBlocks.keys() + theStateAdmin.inputs
        assert all([sy in tmpList for sy in symbs])
        
        self.Y = blockoutputs.next()
        
        theStateAdmin.register_block(self) 
        
        self.expr = expr

    
def exceptionwrapper(fnc):
    """
    prevent the integration algorithm to get stuck if
    a exception occurs in rhs
    """
    
    def newfnc(*args, **kwargs):
        try:
            return fnc(*args, **kwargs)
        except Exception, e:
            import traceback as tb
            tb.print_exc()
            sys.exit(1)
            
    return newfnc
                
#@ this function is too long
def gen_rhs(stateadmin):
    """
    resolves dependencies in the eqns
    and creates a function (called rhs) which can be passed to the
    integration algorithm
    """
    
    subsdict  = {}
    
    # handle integrating blocks (Yii <- SV_jj)
    for y, bl in stateadmin.IBlocks.items():
        subsdict.update({y : stateadmin.stateVars[bl.idcs[0]]})
        
    # handle NIL Blocks
    for y, bl in stateadmin.NILBlocks.items():
        eqn_rhs = stateadmin.get_nil_eq(bl) # can still contain Yii -vars
        subsdict.update({y : eqn_rhs})
        
    # handle Blockfncs
    fncs = {}
    for y, bl in stateadmin.Blockfncs.items():
        new_y = stateadmin.resolve_blockfnc(bl, subsdict)
        fncs[y] = new_y
        
    subsdict.update(fncs)
    
    # now eliminate the Yii vars in the expressions:
    Yii = set(subsdict.keys())
    finished_expr = {}
    
    L0 = len(subsdict)
    for y, expr in subsdict.items():
        if expr.atoms().intersection(Yii) == set():
            finished_expr[y] = expr
            subsdict.pop(y)

    while True:
        L = len(subsdict)
        for y, expr in subsdict.items():
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
        raise ValueError, 'Algebraic loop found (maybe): '+str(L)
    
    for u, expr in loops.items():
        if expr.atoms().intersection(stateadmin.allBlocks) != set():
            raise ValueError, 'unsubstituted expr found: %s=%s' %(u, expr)
        
    
    for y, expr in subsdict.items():
        subsdict[y] = expr.subs(loops)
        
    # we do not need them anymore
    # remove them to avoid confusion
    # but this cause problems if the gen_rhs is called more than once
#    for u in loops:
#        theStateAdmin.inputs.remove(u) 
     
        
    # save the relations for later use
    stateadmin.blockoutdict = subsdict
    
    args = stateadmin.stateVars+stateadmin.inputs
    state_rhs_fncs = []
    
    # the inputs and parameters are taken from global scope (w.r.t. rhs)
    for eq in stateadmin.dynEqns:
        eq = eq.subs(subsdict)
        fnc = sp.lambdify(args, eq)
        state_rhs_fncs.append(fnc)
    
    
    # !! TODO: vectorize and time dependence
    def rhs(z, t, *addargs):
        u = addargs
        fncargs = list(z)+list(addargs)
        dz = [fnc(*fncargs) for fnc in state_rhs_fncs]
        return dz
    
    return exceptionwrapper(rhs)


def compute_block_ouptputs(simresults):
    
    args = theStateAdmin.stateVars + theStateAdmin.inputs
    assert simresults.shape[1] == len(args)
    
    blocks = {}
    
    blockout_vars = set(theStateAdmin.allBlocks)
    
    for v in theStateAdmin.blockoutdict.values():
        # on the right hand side no Yii should be found
        to_be_empty = v.atoms().intersection(blockout_vars)
        
        assert  to_be_empty == set()
    
    for bl in theStateAdmin.allBlocks.values():
        
        y = bl.Y
        # the fnc for calculationg the blockoutput from states
        fnc = sp.lambdify(args, y.subs(theStateAdmin.blockoutdict))
        
        numargs = list(simresults.T)
        # -> list of N 1d-arrays, N=number(statevars)+number(inputs)

        # for each timestep: evaluate the block specific func with the args
        blocks[bl] = [fnc(*na) for na in zip(*numargs)] 
        
    return blocks    


def blocksimulation(tend, inputs = None, z0 = None, dt = 5e-3):
    """
    z0        state for t = 0
    inputs    a dict (like {u1 : calc_u1, u2 : calc_u2} where calc_u1, ... 
              are callables
              (in the case of just one specified input (u1, calc_u1) is allowed  
    """
    
    if z0 == None:
        z0 = [0]*theStateAdmin.dim
    
    assert len(z0) == theStateAdmin.dim
    
    assert float(tend) == tend and tend > 0
    
    if isinstance(inputs, dict):
        pass
    elif inputs == None:
        inputs = {}
    elif hasattr(inputs, '__len__') and len(inputs) == 2 and\
                                    not hasattr(inputs[0], '__len__'):
        inputs = dict([inputs])
    else:
        raise TypeError, "invalid type for input:", str(input)
    
        
    allinputs = {}
    for i in theStateAdmin.inputs:
        allinputs[i] = lambda t: 0 # dummy functions
    
    allinputs.update(inputs)
    inputfncs = [allinputs[i] for i in theStateAdmin.inputs]
    
    t = 0
    z = z0
    
    u0_vect = np.array([fnc(0) for fnc in inputfncs])
    
    stateresults = r_[z0, u0_vect]
    tvect = r_[t]
    
    # generating the model from the blocks
    rhs = gen_rhs(theStateAdmin)
    
    while True:
        u_tup = tuple([fnc(t) for fnc in inputfncs])
        z=integrate.odeint(rhs, z, r_[t,t+dt], u_tup)
        z = z[-1, :]
        stateresults = np.vstack((stateresults, r_[z, u_tup]))
        tvect = np.vstack((tvect, r_[t]))
        
        t+=dt
        
        if t >= tend:
            break
    
    # !! we have two lines with t = 0 in the result
    return tvect, stateresults


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

    def _make_fnc(self, expr, var = None):
        if var == None:
            var = t
        return sp.lambdify( (var,), expr, modules = 'numpy' )
        
    def get_trajectory(self, diff_idx=0):
        assert diff_idx <= self.sd # smoothness_degree
        fnc = self._make_fnc(self.derivatives[diff_idx])
        return fnc

    def combined_trajectories(self, expr):
        """ expects expr to be a polynomial in s which determines
        how to linearily combine the trajectory
        and its derivatives to a new function of time
        
        Example : expr = s**2 -3*s + 5 means

        return a fuction consisting of y_ddot - 3*y_dot + 5*y
        
        """

        coeffs = expr2coeffs(expr, lead_test = False)

        res = 0
        for i,c in enumerate(coeffs):
            res+=c*self.derivatives[i]

        return self._make_fnc(res)


def main():
    pass
    # maybe call an example here?
        

        
theStateAdmin = StateAdmin()

# shortcuts
loop = theStateAdmin.register_loop
inputs = theStateAdmin.register_inputs




if __name__ == '__main__':
    main()