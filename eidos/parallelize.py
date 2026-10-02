"""Helpers for running eidos work across several processes."""

import multiprocessing


def parmap(f, X, proc_power=1):
    """
    A parallelized implementation of the map function. See
    https://www.w3schools.com/python/ref_func_map.asp .

    Parameters
    ----------
    f : function
        Function onto which to map the input arguments in X. It has to be
        importable at module level (i.e. picklable) so that the worker
        processes can run it. Exceptions raised by f are re-raised in the
        caller instead of silently stalling it.
    X : array_like
        The arguments to be fed to the function f. This can only handle a
        single argument for each evaluation of f.
    proc_power : float
        Fraction of the available CPUs to use. A value of 1 or more uses all
        of them.

    Returns
    -------
    out : list
        A list of the outputs for each function evaluation corresponding to the
        input arguments in X.
    """
    nprocs = multiprocessing.cpu_count()
    if 0 < proc_power < 1:
        # Never fall back to zero workers: an empty pool would deadlock.
        nprocs = max(1, int(proc_power * nprocs))

    items = list(X)
    if not items:
        return []

    with multiprocessing.Pool(nprocs) as pool:
        return pool.map(f, items)
