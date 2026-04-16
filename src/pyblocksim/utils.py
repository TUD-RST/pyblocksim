import sympy as sp

def step_factory(y0, y1, x_step=0, tanh_scaling_factor=1e3):
    """
    Factory to create continuously approximated step functions.

    :param y0:       start value of the step function
    :param y1:       final value of the step function
    :param x_step:   x-value at which the step occurs
                     (better: middle point of the step).

    :param tanh_scaling_factor:
                     scaling of shifted x-value, such that the tanh-func
                     becomes more rectangular for bigger values. Default 1e3

    """
    # tanh maps R to (-1, 1)

    # first map R to (0, 1)
    # then map (0, 1) -> (y0, y1)

    dy = y1 - y0

    def fnc(x, module=sp):
        """
        The step function (depends on external variables x_step, dy and y0).

        :param x:       argument of the step function
        :param module:  module object from which to take the tanh-func (numpy or sympy)
        """

        z1 = tanh_scaling_factor * (x - x_step)  # shift and scale x-axis
        z2 = module.tanh(z1)  # map z1 ∈ (-inf, inf) to interval (-1, 1)
        z3 = (z2 + 1) / 2  # map z2 ∈ (-1, 1) to (0, 1)
        z4 = z3 * dy + y0  # map z3 ∈ (0, 1) to (y0, y1)

        return z4

    # save docstring and step parameters
    fnc.__doc__ = "approximated step function %f, %f, %f" % (y0, y1, x_step)
    fnc.x_step = x_step
    fnc.y0 = y0
    fnc.y1 = y1
    fnc.tanh_scaling_factor = tanh_scaling_factor

    return fnc
