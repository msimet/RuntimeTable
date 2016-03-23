import numpy
import scipy.interp

class RuntimeTable(object):
    """
    RuntimeTable: A class to generate interpolation tables at runtime.
    
    RuntimeTable makes multiple calls to the function given as the argument ``func``.  It then
    builds an underlying interpolation table and holds it as one of its attributes (``self.table``).
    If ``func`` has only one variable of interest, it's a ``scipy.interp.interp1d`` table; otherwise
    it's a ``scipy.interp.RegularGridInterpolator`` table.
    
    After creation, you can call this RuntimeTable object like you would have called the
    original ``func``.  If you passed ``*args`` or ``**kwargs`` to this object upon initialization,
    you should NOT pass those same args or kwargs to the ``__call__`` method of this class: they
    were already included when you generated the table!  Multiple arguments should be passed in
    the same order as ``boundaries``.
    
    :param func:                A callable function to generate the RuntimeTable (that is, the 
                                function to be replaced by this RuntimeTable object)
    :param iterable boundaries: The boundaries of the independent variable(s) for the table.  If
                                ``func`` takes one input argument, this can be a 2-item tuple or
                                a one-item list of a 2-item tuple; if ``func`` takes multiple
                                input arguments, this should be a list of 2-item tuples.  Args 
                                passed directly via ``*args`` do not count for this argument count.
    :param nsteps:              The number of steps to take within ``boundaries`` when generating
                                the y-values for the table; alternately, a list of nsteps, one per
                                argument.  If nsteps is iterable this way, then ``boundaries``
                                must also be iterable, with the same length.  If ``boundaries``
                                is iterable but ``nsteps`` is not, the same ``nsteps will be used
                                for every argument.
    :param args:                Any other arguments, which will be passed to func.
    :param kwargs:              Any other keyword arguments, which will be passed to func.
    """
    def __init__(self, func, boundaries=None, nsteps=None, *args, **kwargs):
        if boundaries is None or not hasattr(boundaries, '__iter__'):
            raise RuntimeError("boundaries must be a 2-item tuple describing the range of the "+
                "independent variable, or a list of such tuples for multiple variables!")
        if (nsteps is None or nsteps<=0 or 
         (hasattr(nsteps, '__iter__') and any([n<=0 for n in nsteps]))):
            raise RuntimeError("nsteps must be a positive integer number of steps,"+
                "or a list of such positive integers for multiple variables!")
        if (nsteps!=int(nsteps) or
                hasattr(nsteps, '__iter__') and any([n!=int(n) for n in nsteps])):
            raise RuntimeError("nsteps contains non-integers! %s"%str(nsteps))
        if not hasattr(func, '__call__'):
            raise RuntimeError("func must be a callable function!")
        if hasattr(boundaries[0], '__iter__'):
            if hasattr(nsteps, '__iter__'):
                if not len(boundaries)==len(nsteps):
                    raise RuntimeError("boundaries and nsteps must have the same length, "+
                                       "if both are iterables")
            else:
                nsteps = [nsteps]*len(boundaries)
            if not (all([hasattr(b, '__iter__') for b in boundaries]) and
                    all([len(b)==2 for b in boundaries])):
                raise RuntimeError("Malformed boundaries: %s"%str(boundaries))
            self.n_args = len(boundaries)
        elif hasattr(nsteps, '__iter__'):
            raise RuntimeError('nsteps is iterable, but boundaries is not')
        else:
            self.n_args = 1

        self.args = args
        self.kwargs = kwargs
        self.func = func
        self.boundaries = boundaries
        self.nsteps = nsteps
        
    def SetupTable(self):
        pass
    
    def __call__(self, *args, **kwargs):
        if not len(args)==self.n_args:
            raise ArgumentError("Wrong number of arguments passed to RuntimeTable: expected %i"%
                                self.n_args)
        if self.n_args==1:
            return self.table(*args, **kwargs)
        else:
            return self.table(args, **kwargs)

