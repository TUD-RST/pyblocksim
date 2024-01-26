"""
intermediate module for convenient importing of all needed and useful objects
"""

try:
    # this might fail during installation (which is uncritical)
    from numpy.lib.index_tricks import r_ as r_

    import numpy as np
    import scipy as sc
    import sympy as sp
    import pylab as pl

    from .core import (
        s,
        t,
        TFBlock,
        Blockfnc,
        compute_block_outputs,
        theStateAdmin,
        blocksimulation,
        stepfnc,
        loop,
        inputs,
        Trajectory,
        mainprint,
        sys,
        restart,
        warnings,
        RingBuffer,
        DelayBlock,
        get_linear_ct_model,
        RHSBlock,
    )

    # maintain backward compatibility after fixing typo
    compute_block_ouptputs = compute_block_outputs

except ImportError:
    # this might be relevant during the installation process
    pass
from .release import __version__
