# #############################################################################
# __init__.py
# ===========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

import pathlib

import matplotlib.pyplot as plt
import numpy as np

import opt_tools.util as opt_util


class SolverStat:
    """
    Solver Statistic Interface.

    :py:class:`~opt_tools.solver.SolverStat` contains information from the run
    of :py:class:`~opt_tools.solver.Solver` instances. The output can be
    load/saved from/to disk.

    Examples
    --------
    .. testsetup::

       import numpy as np

       from opt_tools.util import asstructuredarray
       from opt_tools.solver import SolverStat

    .. doctest::

       >>> N, sigma = 50, 0.02
       >>> loss = asstructuredarray(f=np.linspace(1.00, 0.1, N) + sigma * np.random.randn(N),
       ...                          g=np.linspace(0.75, 0.2, N) + sigma * np.random.randn(N),
       ...                          h=np.linspace(0.1, 0.01, N) + sigma * np.random.randn(N))
       >>> data = asstructuredarray(x=np.random.randn(N, 3),  # The optimized variable
       ...                          y=np.random.randn(N, 4))  # Extra info (e.g. dual variable, ...)
       >>> stat = SolverStat(loss, data)

       ax = stat.plot_loss()
       ax.get_figure().show()

    .. image:: _img/solverstat_plot.png
    """

    def __init__(self, loss, data, info=None):
        """
        Parameters
        ----------
        loss : :py:class:`~numpy.ndarray` (float-structured)
            (N,) loss records.

            There can be more than one loss per iteration (i.e., composite cost
            functions.)
        data : :py:class:`~numpy.ndarray` (structured)
            (N,) auxiliary records.

            Interpretation of stored data is
            :py:class:`~opt_tools.solver.Solver`-specific.
        info : str
            Optional supplementary text information.
        """
        if not ((loss.ndim == 1) and
                (loss.dtype.names is not None)):
            raise ValueError('Parameter[loss]: expected 1D structured array.')
        self._loss = loss

        if not ((data.ndim == 1) and
                (data.dtype.names is not None)):
            raise ValueError('Parameter[data]: expected 1D structured array.')
        self._data = data

        if len(loss) != len(data):
            raise ValueError('Parameters[loss, data] have inconsistent shapes.')

        if info is None:
            self._info = ''
        elif isinstance(info, str):
            self._info = info
        else:
            raise ValueError('Parameter[info]: expected string.')

    @property
    def loss(self):
        """
        Returns
        -------
        loss : :py:class:`~numpy.ndarray` (float-structured)
            (N,) loss records.
        """
        return self._loss

    @property
    def data(self):
        """
        Returns
        -------
        data : :py:class:`~numpy.ndarray` (structured)
            (N,) auxiliary records.
        """
        return self._data

    @property
    def info(self):
        """
        Returns
        -------
        info : str
            Optional supplementary text information.
        """
        return self._info

    @classmethod
    def load(cls, path):
        """
        Load information from disk.

        Parameters
        ----------
        path : path-like

        Returns
        -------
        S : :py:class:`~opt_tools.solver.SolverStat`
        """
        path = pathlib.Path(path).expanduser().absolute()
        D = np.load(path)

        S = cls(D['loss'], D['data'], str(D['info']))
        return S

    def save(self, path):
        """
        Save information to disk.

        Parameters
        ----------
        path : path-like
        """
        path = pathlib.Path(path).expanduser().absolute()
        np.savez(path, loss=self._loss,
                       data=self._data,
                       info=np.array(self._info))

    def plot_loss(self, loss=None, ax=None):
        """
        Plot loss functions.

        Parameters
        ----------
        loss : str, collection(str)
            Loss functions to plot.

            If :py:obj:`None`, then plot them all.
        ax : :py:class:`~matplotlib.axes.Axes`
            Axes on which to draw.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if loss is None:
            loss = self._loss.dtype.names
        elif isinstance(loss, str):
            loss = [loss]
        else:
            loss = set(loss)

        N = len(self._loss)
        for l in loss:
            ax.plot(np.arange(N), self._loss[l], label=l)

        if len(loss) > 1:
            cl = np.zeros((N,))  # Composite loss
            for l in loss:
                cl += self._loss[l]
            ax.plot(np.arange(N), cl, label='+'.join(loss))

        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.legend()
        return ax


class Solver:
    pass
