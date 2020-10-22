import numpy as np

def improved_euler_step(f, t, Y, h, pars=[]):
    ''' Return value of dependent variable Y at time (t + h),
        using improved Euler calculation.

        Parameters:
        -----------
        f : callable
            Derivative function.
        t : float
            Current value of independent variable.
        Y : float
            Current value of dependent variable.
        h : float
            Step size.
        pars : array
            Additonal parameters for derivative function, if any.

        Returns:
        --------
        Y1 : float
            value of dependent variable Y at time (t + h).
    '''

    # Current gradient
    dY0 = f(t, Y, *pars)

    # Euler step
    Y1 = Y + h * dY0

    # Predictor gradient
    dY1 = f(t + h, Y1, *pars)

    # corrector step
    return Y + h * (dY0 + dY1) / 2


def improved_euler_solve(f, ts, tf, Y0, h, pars=[]):
    ''' Returns time vector and respective solution at time vector
        for a derivative function using improved Euler method.

        Parameters:
        -----------
        f : callable
            Derivative function.
        ts : float
            Start time of time vector.
        tf : float
            End of time vector.
        Y0 : float
            Initial value of dependent variable.
        h : float
            Step size.
        pars : array
            Additonal parameters for derivative function, if any.

        Returns:
        --------
        t : np.array
            Time vector from ts to tf with step size h.
        Y : np.array
            Solution vector for derivative function f and time vector t.
    '''
    # create time vector
    t = np.array([])
    while ts <= tf:
        t = np.append(t, ts)
        ts += h

    # solve for the value of Y at each value in the time vector
    Y = np.array([Y0])
    for i in range(len(t) - 1):
        Y = np.append(Y, improved_euler_step(f, t[i], Y[i], h, pars))
    
    return t, Y
