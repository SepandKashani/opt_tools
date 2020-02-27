# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

import abc


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
