# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

import numpy as np


def asstructuredarray(**kwargs):
    """
    Pack arguments into 1D structured array.

    Parameters
    ----------
    kwargs.values[k] : :py:class:`~numpy.ndarray`
        (N, [...]) data to pack.

        The various `kwargs.values[k]` can have different shapes, but their
        first dimension must coincide.

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N,) structured array.

        `S[kwargs.key[k]]` = `kwargs.values[k]`

    Examples
    --------
    .. testsetup::

       import numpy as np

       from opt_tools.util import asstructuredarray

    .. doctest::

       >>> N = 2
       >>> A = np.arange(N * 2, dtype=int).reshape((N, 2))
       >>> B = np.arange(N * 4 * 5, dtype=float).reshape((N, 4, 5))
       >>> S = asstructuredarray(A=A, B=B)

       >>> np.allclose(S['A'], A), np.allclose(S['B'], B)
       (True, True)
    """
    sh = [_.shape for _ in kwargs.values()]
    N = sh[0][0]
    if not all([N == _[0] for _ in sh]):
        raise ValueError('Parameters differ in their first dimension.')

    dS, S = [], []
    for k, v in kwargs.items():
        dS.append((k, v.dtype, v.shape[1:]))
        S.append(np.ascontiguousarray(v)
                 .reshape((N, -1))
                 .view(np.uint8))

    dS = np.dtype(dS)
    S = (np.concatenate(S, axis=-1)
         .view(dS)
         .reshape((N,)))
    return S


