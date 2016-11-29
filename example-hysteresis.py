# -*- coding: utf-8 -*-

from pyblocksim import *
#import numpy as np


print("""
Approximation of a simple hysteresis system
""")


# switching tresholds for hysteresis
x1 = 4  # switching down
x2 = 8  # switching up


# output values of the hysteresis system
y1 = 2
y2 = 11

# time constant of the internal PT1
T_storage = 1e-4


_tanh_factor = 1e3

def step_factory(y0, y1, x_step):
    """
    Factory to create continously approximated step functions
    """
    #tanh maps R to (-1, 1)
    
    # first map R to (0, 1)
    # then map (0, 1) -> (y0, y1)
    
    dy = y1-y0
    
    def fnc(x, module=sp):
     return (module.tanh(_tanh_factor*(x-x_step))+1 )/2*dy + y0
 
    fnc.__doc__ = "approximated step function %f, %f, %f" % (y0, y1, x_step)
 
    return fnc


#  togehter these two are applied to the input
step1 = step_factory(-1, 0, x1)
step2 = step_factory(0, 1, x2)

xx = np.linspace(1, 10, 1e5)

#pl.plot(xx, step1(xx, np))
#pl.plot(xx, step2(xx, np) + .1)



# the third step-function limits the input of the PT1 between between 0 and 1
# the exact value for 63 % Percent
time_const_value = 1 - np.exp(-1)
step3 = step_factory(0, 1, time_const_value)




hyst_in, fb = inputs('hyst_in, fb') # overall input and internal feedback


# the sum of the two step-functions is basically a three-point-controller
SUM1 = Blockfnc(step1(hyst_in) + step2(hyst_in) + fb)

LIMITER = Blockfnc(step3(SUM1.Y))

PT1_storage = TFBlock(1/(T_storage*s + 1), LIMITER.Y)

loop(PT1_storage.Y, fb)

# gain and offset for the output
Result = Blockfnc(PT1_storage.Y*(y2-y1) + y1)


def input_ramps(t):
    
    T1 = 10
    T2 = 20
    k1 = 1
    k2 = 1
    
    if t < 0:
        return 0
    elif t < T1:
        return k1*t
    elif t < T2:
        return k1*T1 - k2*(t-T1)
    else:
        return 0
    
    

tt, states = blocksimulation(25, {hyst_in: input_ramps}) # simulate
tt = tt.flatten()

bo = compute_block_ouptputs(states)


pl.plot(tt, [input_ramps(t) for t in tt], label='input')
pl.plot(tt, bo[Result], label='hyst. output')

pl.grid(1)
pl.legend()

pl.show()

quit()
