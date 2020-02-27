# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

import abc

import numpy as np
import numpy.linalg as linalg


class Function(abc.ABC):
    r"""
    Function Interface.

    :py:class:`~opt_tools.function.Function` encodes mathematical functions
    :math:`f : \mathbb{C}^{\texttt{sh}_{I}} \to \mathbb{R}`.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, x):
        r"""
        Evaluate function.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (\*sh_I,) input array (real/complex)

        Returns
        -------
        y : float
            Function value:

            :math:`y = f(x)`
        """
        raise NotImplementedError


class ProximableFunction(Function):
    r"""
    Proximable Function Interface.

    :py:class:`~opt_tools.function.ProximableFunction` encodes proximable
    mathematical functions :math:`f : \mathbb{C}^{\texttt{sh}_{I}} \to \mathbb{R}.`
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def prox(self, x, lambda_=1):
        r"""
        Proximal operator (not necessarily canonical).

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (\*sh_I,) input array (real/complex)
        lambda_ : float

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            (\*sh_I,) proximal point:

            :math:`y = \prox_{\lambda f}(x) = \arg\min_{u \in \mathbb{C}^{\texttt{sh}_{I}}} 2 \lambda f(u) + \|x - u\|_{q}^{2}`,
            where :math:`\|\cdot\|_{q}` can be any (valid) norm.
        """
        raise NotImplementedError

    def prox_c(self, x, lambda_=1):
        r"""
        Proximal operator of convex conjugate.

        .. math::

            \prox_{\lambda f^{*}}(x) = x - \lambda \prox_{\frac{1}{\lambda} f}\left(\frac{x}{\lambda}\right)

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (\*sh_I,) input array (real/complex)
        lambda_ : float

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            (\*sh_I,) proximal point:

            :math:`y = \prox_{\lambda f^{*}}(x) = \arg\min_{u \in \mathbb{C}^{\texttt{sh}_{I}}} 2 \lambda f^{*}(u) + \|x - u\|_{q}^{2}`,
            where :math:`\|\cdot\|_{q}` can be any (valid) norm.
        """
        y = x - lambda_ * self.prox(x / lambda_, 1 / lambda_)
        return y


class DifferentiableFunction(Function):
    r"""
    Differentiable Function Interface.

    :py:class:`~opt_tools.function.DifferentiableFunction` encodes differentiable
    mathematical functions :math:`f : \mathbb{C}^{\texttt{sh}_{I}} \to \mathbb{R}.`
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def grad(self, x):
        r"""
        Gradient operator.

        Parameters
        ----------
        x : :py:class:`~numpy.ndarray`
            (\*sh_I,) input array (real/complex)

        Returns
        -------
        y : :py:class:`~numpy.ndarray`
            (\*sh_I,) gradient:

            :math:`y = \nabla f(x)`.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def g_lipschitz(self):
        r"""
        A Lipschitz constant :math:`L \in [0, \infty[` of the gradient such that:

        :math:`\|\nabla f(x) - \nabla f(y) \|_{2}^{2} \le L \|x - y\|_{2}^{2}, \quad \forall (x, y) \in \mathbb{C}^{\texttt{sh}_{I}}`.

        Returns
        -------
        L : float
            Lipschitz constant.
        """
        raise NotImplementedError


class Loss_L2(DifferentiableFunction, ProximableFunction):
    r"""
    :math:`f(x) = \|y - A x\|_{F}^{2}`

    With:

    * :math:`x \in \mathbb{C}^{\texttt{sh}_{I}}`;

    * :math:`y \in \mathbb{C}^{\texttt{sh}_{O}}`;

    * :math:`A \in \mathbb{C}^{\texttt{sh}_{O} \times \texttt{sh}_{I}}`;

    * :math:`\|z\|_{F}^{2} = \|\vecop(z)\|_{2}^{2}`.

    Notes
    -----
    * This implementation assumes :math:`A` is dense, so computations take place
      internally via 2D arrays.
    * :py:meth:`~opt_tools.function.Loss_L2.prox` and
      :py:meth:`~opt_tools.function.Loss_L2.g_lipschitz` are computed via
      :math:`\texttt{SVD}(A)`.
    """

    def __init__(self, A, y):
        r"""
        Parameters
        ----------
        A : :py:class:`~numpy.ndarray`
            (\*sh_O, \*sh_I) array (real/complex)
        y : :py:class:`~numpy.ndarray`
            (\*sh_O,) array (real/complex)
        """
        self._sh_O = A.shape[:y.ndim]
        self._sh_I = A.shape[y.ndim:]
        self._y = np.reshape(y, (np.prod(self._sh_O),))
        self._A = np.reshape(A, (np.prod(self._sh_O), np.prod(self._sh_I)))
        self._AH = self._A.T.conj()

        # Some precomputations for speed
        self._S = None
        self._V = None
        self._VH = None
        self._glipschitz = None

    def __call__(self, x):
        y = linalg.norm(self._y - self._A @ x.reshape(-1)) ** 2
        return y

    def grad(self, x):
        y = self._AH @ (self._A @ x.reshape(-1) - self._y)
        return y.reshape(self._sh_I)

    @property
    def g_lipschitz(self):
        if self._glipschitz is None:
            # The lower bound for `L` is \sigma_{\max}^{2}(A)
            _, self._S, self._VH = linalg.svd(self._A, full_matrices=False)
            self._V = self._VH.T.conj()
            self._glipschitz = np.amax(self._S) ** 2

        L = self._glipschitz
        return L

    def prox(self, x, lambda_=1):
        # \prox_{\lambda f}(x) = (I + 2 \lambda A^{H}A)^{-1} (2 \lambda A^{H} y + x)
        #                      = V [I + 2 \lambda S^{H} S]^{-1} V^{H} (2 \lambda A^{H} y + x)
        if np.isclose(lambda_, 0):
            y = x
        else:
            if self._S is None:
                _, self._S, self._VH = linalg.svd(self._A, full_matrices=False)
                self._V = self._VH.T.conj()

            a = (self._AH @ self._y) + (x.reshape(-1) / (2 * lambda_))
            b = (self._S ** 2) + (1 / (2 * lambda_))

            y = np.reshape((self._V / b) @ (self._VH @ a), self._sh_I)

        return y
