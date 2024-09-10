"""
intermediate module for convenient importing of all needed and useful objects
"""

try:
    # this might fail during installation (which is uncritical)
    from numpy import r_

    import numpy as np
    import scipy as sc
    import sympy as sp

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
        HyperBlock,
        restart,
    )
    from . import td

    # maintain backward compatibility after fixing typo
    compute_block_ouptputs = compute_block_outputs

except ImportError as ex:
    # this might be relevant during the installation process
    print(ex)
    raise
from .release import __version__
