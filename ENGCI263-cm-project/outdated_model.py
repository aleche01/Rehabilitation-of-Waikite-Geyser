from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import numpy as np
from scipy.optimize import curve_fit

######### OUTDATED MODEL #########

# get data
# pressure
tP_data, w_data = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T
P_data = (101325 + 1000 * 9.81 * w_data / 2)
# temperature
tT_data, T_data = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T

# get and interpolate q_data and dq_data
qt_data, q_data = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T
q_interp = np.interp(tP_data, qt_data, q_data)
dq_data = 0 * q_data                                                            # allocate derivative vector
dq_data[1:-1] = (q_data[2:] - q_data[:-2]) / (qt_data[2:] - qt_data[:-2])       # central differences
dq_data[0] = (q_data[1] - q_data[0]) / (qt_data[1] - qt_data[0])                # forward difference
dq_data[-1] = (q_data[-1] - q_data[-2])/(qt_data[-1] - qt_data[-2])             # backward difference


def pressure_ode(t, P, ap, bp, cp, P0):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent variable.
        ap : float
            Extraction rate strength parameter.
        bp : float
            Recharge strength parameter. 
        cp : float
            Slow drainage strength parameter. 
        P0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dPdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
    
    # get extraction rate at time t (via interpolation)
    qt = np.interp(t, qt_data, q_data)
    dqt = np.interp(t, qt_data, dq_data)
    return -ap * qt - bp * (P - P0) - cp * dqt


def temperature_ode(t, T, aT, bT, Tc, T0):
    ''' Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        aT : float
            Cold water inflow strength parameter.
        bT : float
            Conduction strength parameter. 
        Tc : float
            Temperature of cold water.
        T0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

    '''
    
    # get pressure at time t (via interpolation)
    Pt = np.interp(t, tP_model, P_model)
    return -aT * (bp / ap) * (Pt - P0) * (Tc - T) - bT * (T - T0)


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
    # current gradient
    dY0 = f(t, Y, *pars)

    # Euler step
    Y1 = Y + h * dY0
    # predictor gradient
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
    while ts < tf:
        t = np.append(t, ts)
        ts += h

    # solve for the value of Y at each value in the time vector
    Y = np.array([Y0])
    for i in range(len(t) - 1):
        Y = np.append(Y, improved_euler_step(f, t[i], Y[i], h, pars))
    
    return t, Y


def solve_pressure(t, ap, bp, cp, P0, Pi):
    ''' Solves pressure_ode for given time vector and parameters.
        Uses improved Euler method.

        Parameters:
        -----------
        t : array
            Time vector to solve derivative function for.
        ap : float
            Extraction rate strength parameter.
        bp : float
            Recharge strength parameter. 
        cp : float
            Slow drainage strength parameter. 
        P0 : float
            Ambient value of dependent variable.
        Pi : float
            initial value of P at t[0]

        Returns:
        --------
        Y : np.array
            Solution vector for time vector t.
    '''
    h = 0.1
    tP_solved, P_solved = improved_euler_solve(pressure_ode, t[0], t[-1], Pi, h, [ap, bp, cp, P0])
    return np.interp(t, tP_solved, P_solved)


def solve_temperature(t, aT, bT, Tc, T0, Ti):
    ''' Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        aT : float
            Cold water inflow strength parameter.
        bT : float
            Conduction strength parameter. 
        Tc : float
            Temperature of cold water.
        T0 : float
            Ambient value of dependent variable.
        Ti : float
            initial value of T at t[0]

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

    '''
    h = 0.1
    tT_solved, T_solved = improved_euler_solve(temperature_ode, t[0], t[-1], Ti, h, [aT, bT, Tc, T0])
    return np.interp(t, tT_solved, T_solved)

# small step size
h = 0.1

# pressure model
# estimates parameters (taken from previous calibration, makes it run faster)
P_params_estimates = [1.94982339e-01, 9.58103862e-02, 1.09031740e+00, 1.57837642e+06, 1.54750029e+06] #<-- use this if /2
# P_params_estimates = [3.89980416e-01, 9.58148505e-02, 2.18063115e+00, 3.05542746e+06, 2.99367563e+06] #<-- use this if not / 2
# gets parameters and covariance via calibration
P_params, P_covs = curve_fit(solve_pressure, tP_data, P_data, p0=P_params_estimates)
print('parameters for pressure: ', P_params)
ap = P_params[0]
bp = P_params[1]
cp = P_params[2]
P0 = P_params[3]
# Pi = P_params[4] <-- no longer need this
P_params = P_params[:-1]

# runs pressure model from beginning to end of extraction rate, initial pressure P0 (ambient value)
tP0 = qt_data[0]
tP1 = qt_data[-1]
tP_model, P_model = improved_euler_solve(pressure_ode, tP0, tP1, P0, h, P_params)

# plots pressure model against data
f, axP = plt.subplots()
axP.plot(tP_data, P_data / 1e6, 'ko', label='data')
axP.plot(tP_model, P_model / 1e6, 'b--', label='model solution')
axP.set_title('Old Calibrated Pressure Model vs Data')
axP.set_xlabel('time (year)')
axP.set_ylabel('pressure (MPa)')
axP.legend()
plt.savefig('plots/pressure_model_old', bbox_inches='tight')

# plots misfit for pressure model
f, axPm = plt.subplots()
P_misfit = np.interp(tP_data, tP_model, P_model) - P_data
axPm.plot(tP_data, P_misfit, 'bx')
axPm.set_title('Old Misfit of Pressure Model Against Data')
axPm.set_xlabel('time (years)')
axPm.set_ylabel('pressure (Pa)')
axPm.axhline(y=0, linestyle='dotted', color='k')
plt.savefig('plots/pressure_misfit_old', bbox_inches='tight')

# temperature model
# T_params_guesses = [1.e-4, 1, 75, 150, 140] #<-- use these in case other estimates are faulty
# estimates parameters (taken from previous calibration, makes it run faster)
T_params_guesses = [5.13299333e-06, 3.25895469e-01, 6.25000000e+01, 1.50595634e+02, 1.48478510e+02] #<-- use this if /2
# T_params_guesses = [5.13293295e-06, 3.25891403e-01, 6.25000000e+01, 1.50595695e+02, 1.48478495e+02] #<-- use this if not /2
# set bounds so 62.5 <= Tc <= 87.5, read it somewhere in the literature that Tc ~= 75 deg C
T_bounds = ([-np.inf, -np.inf, 62.5, -np.inf, -np.inf], [np.inf, np.inf, 87.5, np.inf, np.inf])
# gets parameters and covariance via calibration
T_params, T_covs = curve_fit(solve_temperature, tT_data, T_data, p0=T_params_guesses, bounds=T_bounds, maxfev=20000)
print('parameters for temperature: ', T_params)
aT = T_params[0]
bT = T_params[1]
Tc = T_params[2]
T0 = T_params[3]
# Ti = T_params[4] <-- no longer need this
T_params = T_params[:-1]

# runs temperature model from beginning to end of extraction rate, initial temperature T0 (ambient value)
tT0 = tP0
tT1 = tP1
tT_model, T_model = improved_euler_solve(temperature_ode, tP0, tP1, T0, h, T_params)

# plots temperature model against data
f, axT = plt.subplots()
axT.plot(tT_data, T_data, 'ko', label='data')
axT.plot(tT_model, T_model, 'r--', label='model solution')
axT.set_title('Old Calibrated Temperature Model vs Data')
axT.set_xlabel('time (year)')
axT.set_ylabel('temperature ({}C)'.format(chr(176)))
axT.legend()
plt.savefig('plots/temperature_model_old', bbox_inches='tight')

# plots misfit for temperature model
f, axTm = plt.subplots()
T_misfit = np.interp(tT_data, tT_model, T_model) - T_data
axTm.plot(tT_data, T_misfit, 'rx')
axTm.set_title('Old Misfit of Temperature Model Against Data')
axTm.set_xlabel('time (years)')
axTm.set_ylabel('temperature ({}C)'.format(chr(176)))
axTm.axhline(y=0, linestyle='dotted', color='k')
plt.savefig('plots/temperature_misfit_old', bbox_inches='tight')
