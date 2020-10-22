from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import matplotlib.lines as mlines
import numpy as np
from scipy.optimize import curve_fit
from improved_euler_functions import *

########## Data generation ##########

def generate_data():
    ''' 
    Generate all supplied extraction/temperature/pressure data.

    Parameters:
    -----------
    -
    Returns:
    --------
    tP_data : array-like
        time over which pressure data was collected
    P_data : array-like
        pressure data collected from near Whakarewarewa
    tT_data : array-like
        time over which temperature data was collected
    T_data : array-like
        temperature data collected from near Whakarewarewa
    qt_data : array-like
        time over which extraction data outside rhyolite formation was collected
    qt_data_total : array-like
        time over which total extraction rate data was collected
    q_data : array-like
        extraction rate data from outside rhyolite formation
    q_data_total : array-like
        total extraction rate data from Rotorua geothermal system
    dq_data : array-like
        extraction rate derivative data
    tR_data : array-like
        time over which equilibrium temp data was collected
    R_data : array-like
        equilibrium temperature
    '''

    # Sets data to be global variables so they can be accessed by functions they are not easily able to
    # be passed into
    global tP_data, P_data, tT_data, T_data, qt_data, q_data, dq_data, tR_data, R_data

    # Pressure data
    tP_data, w_data = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T
    # Conversion to hydrostatic pressure in middle of reservoir
    P_data = (101325 + 1000 * 9.81 * w_data / 2)

    # Temperature data
    tT_data, T_data = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T

    # Extraction rate data
    qt_data_total, q_data_total = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T
    qt_data_rhy, q_data_rhy = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T

    # Generate extraction rate data for non-rhyolite areas
    qt_data = np.linspace(1950, 2015, 651)
    q_data = np.interp(qt_data, qt_data_total, q_data_total) - np.interp(qt_data, qt_data_rhy, q_data_rhy)

    # Extraction rate derivative data
    dq_data = 0 * q_data
    dq_data[1:-1] = (q_data[2:] - q_data[:-2]) / (qt_data[2:] - qt_data[:-2])       # central differences
    dq_data[0] = (q_data[1] - q_data[0]) / (qt_data[1] - qt_data[0])                # forward difference
    dq_data[-1] = (q_data[-1] - q_data[-2])/(qt_data[-1] - qt_data[-2])             # backward difference

    # equilibrium temp data
    tR_data = [1967, 1967, 1967]
    R_data = [100, 100, 100]

    # Universal step size for all model functions
    global h
    h = 0.1

    return tP_data, P_data, tT_data, T_data, qt_data, qt_data_total, q_data, q_data_total, dq_data, tR_data, R_data

########## Model calibration ##########

def pressure_ode(t, P, ap, bp, cp, P0):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Time.
        P : float
            Pressure at time t.
        ap : float
            Extraction rate strength parameter.
        bp : float
            Recharge strength parameter. 
        cp : float
            Slow drainage strength parameter. 
        P0 : float
            Ambient/intial value of pressure.

        Returns:
        --------
        dPdt : float
            Rate of change of pressure at given time.
    '''
    
    # Interpolate value of extraction rate/derivative at given time
    qt = np.interp(t, qt_data, q_data)
    dqdt = np.interp(t, qt_data, dq_data)

    # Return derivative at specified time
    return -ap * qt - bp * (P - P0) - cp * dqdt

def temperature_ode(t, T, aT, bT, T0):
    ''' Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Time.
        T : float
            Temperature at given time.
        aT : float
            Cold water inflow strength parameter.
        bT : float
            Conduction strength parameter. 
        T0 : float
            Ambient/initial value of temperature.

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

    '''
    
    # Interpolate pressure at given time
    Pt = np.interp(t, tP_model, P_model)

    # Return derivative at required time
    if (Pt > P_params[3]):
        return - bT * (T - T0)
    else:
        return -aT * (P_params[1] / P_params[0]) * (Pt - P_params[3]) * (30 - T) - bT * (T - T0)

def recovery_condition(t, qc, k, Pc):
    
    ''' Calculates the recovery condition at times, t, for given parameters.

        Parameters:
        -----------
        t : array
            Time vector to solve recovery condition for.
        qc : float
            Extraction rate of cold water.
        k : float
            Proportionality constant for extraction rate of hot water.
        Pc : float
            Critical pressure value.

        Returns:
        --------
        R_model : np.array
            Recovery condition values at times t.
    '''

    # Interpolate relevant values of pressure and temperature 
    P = np.interp(t, tP_model, P_model)
    T = np.interp(t, tP_model, T_model)

    # Calulate equilibrium temperature vector
    return (30*qc + T*k*(P - Pc)) / (qc + k*(P - Pc))

def solve_pressure(t, ap, bp, cp, P0):
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

    # Solve pressure ode using improved Euler method
    tP_solved, P_solved = improved_euler_solve(pressure_ode, qt_data[0], qt_data[-1], P0, h, [ap, bp, cp, P0])

    # Return solution at given time points
    return np.interp(t, tP_solved, P_solved)

def solve_temperature(t, aT, bT, T0):
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
        T0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

    '''

    # Solve temperature ode using improved Euler method.
    tT_solved, T_solved = improved_euler_solve(temperature_ode, qt_data[0], qt_data[-1], T0, h, [aT, bT, T0])
    
    # Return solution at given time points.
    return np.interp(t, tT_solved, T_solved)

def calibrate_pressure_model(tP_data, P_data):
    ''' 
    Calibrates pressure model to data

    Parameters:
    -----------
    tP_data : array-like
        Time values of pressure data
    P_data : array-like
        Pressure data to be calibrated to

    Returns:
    --------
    P_params
        Values of parameters of best fit pressure model
    P_covs : array-like
        Covariance matrix for pressure model
    tP_model : array-like
        Time values of final model
    P_model : array-like
        Values of pressure model at given time values
    '''
    
    # Make model and model parameters global so they can be accessed by functions it is difficult to pass them into directly.
    global tP_model, P_model, P_params
    
    # Estimates parameters (increases efficiency and helps to avoid local minima)
    P_params_estimates = [1.53674583e-01, 1.15790160e-01, 6.20234384e-01, 1.56033496e+06]
    
    # Calibrate to obtain parameters/covariance matrix
    P_params, P_covs = curve_fit(solve_pressure, tP_data, P_data, p0=P_params_estimates, sigma=[800]*len(P_data), absolute_sigma=True)
    
    # Generate model
    tP_model, P_model = improved_euler_solve(pressure_ode, qt_data[0], qt_data[-1], P_params[3], 0.1, P_params)

    print('Parameters of pressure best fit model:', P_params)

    return P_params, P_covs, tP_model, P_model

def calibrate_temperature_model(tT_data, T_data):
    ''' 
    Calibrates temperature model to data

    Parameters:
    -----------
    tT_data : array-like
        Time values of temperature data
    T_data : array-like
        Temperature data to be calibrated to

    Returns:
    --------
    T_params
        Values of parameters of best fit temperature model
    T_covs : array-like
        Covariance matrix for temperature model
    tT_model : array-like
        Time values of final model
    T_model : array-like
        Values of temperature model at given time values
    '''
    
    global tT_model, T_model, T_params
    
    # Estimates parameters (taken from previous calibration, makes it run faster and helps to avoid local minima)
    T_params_guesses = [1.90235605e-06, 8.04515747e-02, 1.49045057e+02]
    
    # Calibrates to obtain parameters/covariance
    T_params, T_covs = curve_fit(solve_temperature, tT_data, T_data, p0=T_params_guesses, maxfev=20000, sigma=[1]*len(T_data), absolute_sigma=True)
    
    # Generate model
    tT_model, T_model = improved_euler_solve(temperature_ode, qt_data[0], qt_data[-1], T_params[2], h, T_params)
    
    print('Parameters for temperature best fit:', T_params)

    return T_params, T_covs, tT_model, T_model

def calibrate_recovery_condition(tR_data, R_data):
    ''' 
    Calibrates recovery model to data

    Parameters:
    -----------
    tR_data : array-like
        Time at which geyser last erupted
    R_data : array-like
        Equilibrium temperature at time at which it last erupted

    Returns:
    --------
    R_params
        Values of parameters of best fit equilibrium temperature model
    R_covs : array-like
        Covariance matrix for equilibrium temperature model
    tR_model : array-like
        Time values of final model
    R_model : array-like
        Values of equilibrium temperature model at given time values
    '''
    

    # Same time vector as pressure model
    tR_model = tP_model

    # Initial guesses and bounds for parameters
    R_params_guesses = [1e+05, 1e1, 1.01000001e+06]
    R_bounds = ([-np.inf, 1e1, 1e6], [np.inf, np.inf, np.inf])

    # Calibrate model to data
    R_params, R_covs = curve_fit(recovery_condition, tR_data, R_data, maxfev=10000000000, p0=R_params_guesses, bounds=R_bounds, sigma=[0.8]*len(R_data), absolute_sigma=True)
    
    print('Parameters of recovery condition best fit model:', R_params)

    # Generate calibrated model
    R_model = recovery_condition(tR_model, R_params[0], R_params[1], R_params[2])
    print('Final value of recovery model:', R_model[-1])

    return R_params, R_covs, tR_model, R_model

def calibrate_recovery_condition_alt(tR_data, R_data):
    ''' 
    Calibrates recovery model to data (alternative version for illustrative purposes in the report.)

    Parameters:
    -----------
    tR_data : array-like
        Time at which geyser last erupted
    R_data : array-like
        Equilibrium temperature of geyser at time at which it last erupted

    Returns:
    --------
    R_params
        Values of parameters of best fit equilibrium temperature model
    R_covs : array-like
        Covariance matrix for inflow temperature model
    tR_model : array-like
        Time values of final model
    R_model : array-like
        Values of equilibrium temperature model at given time values
    '''
    

    # Same time vector as pressure model
    tR_model = tP_model

    # Initial guesses and bounds for parameters
    R_params_guesses = [1e+05, 1e1, 1.e4]
    R_bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])

    # Calibrate model to data
    R_params, R_covs = curve_fit(recovery_condition, tR_data, R_data, maxfev=10000000000, p0=R_params_guesses, bounds=R_bounds, sigma=[0.8]*len(R_data), absolute_sigma=True)
    
    print('Parameters of recovery condition best fit model (alt):', R_params)

    # Generate calibrated model
    R_model = recovery_condition(tR_model, R_params[0], R_params[1], R_params[2])
    print('Final value of alt model:', R_model[-1])

    return R_params, R_covs, tR_model, R_model

########## Scenario modelling and forecasts ##########

def pressure_scenario_model():
    '''
    Models resultant reservoir pressure after four different extraction rate scenarios.

    Parameters:
    -----------
    -

    Returns:
    --------
    tP_model_s : array-like
        time vector of scenarios
    P_model_s : array-like
        contains modelled pressure data after each scenario

    '''
    
    # Allows extraction rate data to be accessed and modified
    global q_data
    global qt_data
    global dq_data

    # Vector for scenario data
    P_model_s = [[], [], [], []]

    # Adds extra data to extraction rate data
    qt_data = np.concatenate([qt_data, [2014.9, 2050]])
    q_data = np.concatenate([q_data, [0, 0]])
    dq_data = np.concatenate([dq_data, [0, 0]])

    # Extraction rates for each scenario
    extraction_rates = [0,500,250,1000]

    # Solves model for each scenario
    for i in range(4):
        q_data[-2:] = [extraction_rates[i], extraction_rates[i]]
        tP_model_s, P_model_s[i] = improved_euler_solve(pressure_ode, qt_data[0], qt_data[-1], P_params[3], h, P_params)
        print('Scenario', i+1, '2050 pressure [MPa]:', round(P_model_s[i][-1]/1e6, 5))

    # Restores extraction rate data to former values
    [q_data, qt_data, dq_data] = [q_data[:-2], qt_data[:-2], dq_data[:-2]] 

    return tP_model_s, P_model_s

def temperature_scenario_model(tP_model_s, P_model_s):
    '''
    Models resultant reservoir temperature after four different extraction rate scenarios.

    Parameters:
    -----------
    tP_model_s : array-like
        time vector of scenarios
    P_model_s : array-like
        contains modelled pressure data after each scenario

    Returns:
    --------
    tT_model_s : array-like
        time vector of scenarios
    T_model_s : array-like
        contains modelled temperature data after each scenario
    '''
    
    global tP_model, P_model, qt_data, q_data, dq_data

    # Adds relevant values to extraction rate data
    qt_data = np.concatenate([qt_data, [2014.9, 2050]])
    q_data = np.concatenate([q_data, [0, 0]])
    dq_data = np.concatenate([dq_data, [0, 0]])

    # Vector for scenario data
    T_model_s = [[], [], [], []]

    tP_model = tP_model_s

    # Solves model for each scenario
    for i in range(4):
        P_model = P_model_s[i]
        tT_model_s, T_model_s[i] = improved_euler_solve(temperature_ode, qt_data[0], qt_data[-1], T_params[2], h, T_params)
        print('Scenario', i+1, '2050 temperature [deg C]:', round(T_model_s[i][-1], 3))

    # Reset pressure model
    [tP_model, P_model] = [tP_model[:651], P_model[:651]]
    [q_data, qt_data, dq_data] = [q_data[:-2], qt_data[:-2], dq_data[:-2]] 

    return tT_model_s, T_model_s

def recovery_scenario_model(tP_model_s, P_model_s, T_model_s, R_params):
    '''
    Models recovery condition after four different extraction rate scenarios.

    Parameters:
    -----------
    tP_model_s : array-like
        time vector of scenarios
    P_model_s : array-like
        contains modelled pressure data after each scenario
    T_model_s : array-like
        contains modelled temperature data after each scenario
    R_params : array-like
        contains parameters for recovery model.

    Returns:
    --------
    tR_model_s : array-like
        time vector of scenarios
    R_model_s : array-like
        contains modelled recovery condition data after each scenario
    '''
    
    global P_model, T_model, tP_model

    tR_model_s = tP_model_s
    R_model_s = [[], [], [], []]
    tP_model = tP_model_s
    
    # For each scenario, calculate equilibrium temperature using relevant pressure and temperature models
    for i in range(4):
        P_model = P_model_s[i]
        T_model = T_model_s[i]
        R_model_s[i] = recovery_condition(tP_model, R_params[0], R_params[1], R_params[2])
        print('Scenario', i+1, '2050 equilibrium temperature [deg C]:', round(R_model_s[i][-1], 2))
        # Determine time at which equilibrium temperature first exceeds 100 deg C
        for j in range(len(R_model_s[i])):
            if (R_model_s[i][j] >= 100) & (tR_model_s[j] > 1970):
                print('Scenario', i+1, 'year in which equilibrium temperature >= 100 deg C:', round(tR_model_s[j], 0))
                break
  
    # Reset pressure and temperature models
    P_model = P_model[:651]
    tP_model = tP_model[:651]
    T_model = T_model[:651]

    return tR_model_s, R_model_s

def pressure_forecast(P_covs, P_params, n):
    '''
    Forecasts reservoir pressure after four different extraction rate scenarios, taking uncertainty into account

    Parameters:
    -----------
    tP_covs : array-like
        covariance matrix of pressure model
    P_params : array-like
        parameters of best fit pressure model

    Returns:
    --------
    uncertainty_data_pressure : array-like
        modelled pressure after each scenario, taking uncertainty into account
    n : int
        number of samples of covariance matrix to take

    '''

    global q_data, qt_data, dq_data

    # Vector to contain uncertainty data
    uncertainty_model = [[], [], [], []]

    # Extraction rates for different scenarios
    extraction_rates = [0, 500, 250, 1000]

    # Extends extraction rate data
    [q_data, qt_data, dq_data] = [np.concatenate([q_data, [0, 0]]), np.concatenate([qt_data, [2014.6, 2050]]), np.concatenate([dq_data, [0, 0]])]

    # Pressure vector for each uncertainty scenario
    pressure_2050 = [[], [], [], []]

    # Runs through combinations of parameters sampled from the posterior and calculates pressure model
    ps = np.random.multivariate_normal(P_params, P_covs, n)
    for pi in ps:
        for i in range(4):
            q_data[-2:] = [extraction_rates[i], extraction_rates[i]]
            _, P_model_uncertainty = improved_euler_solve(pressure_ode, qt_data[0], qt_data[-1], pi[3], h, pi)
            uncertainty_model[i].append(P_model_uncertainty)
            pressure_2050[i].append(P_model_uncertainty[-1])
    
    # Resets extraction rate data
    [q_data, qt_data, dq_data] = [q_data[:-2], qt_data[:-2], dq_data[:-2]] 

    for i in range(4):
        print('Scenario', i+1, '2050 pressure 5/95% confidence interval [MPa]:', np.round(np.quantile(pressure_2050[i], [0.05, 0.95])/1e6, 5))
        
    return uncertainty_model

def temperature_forecast(T_covs, T_params, tP_model_s, P_model_s, n):
    
    '''
    Forecasts reservoir temperature after four different extraction rate scenarios, taking uncertainty into account

    Parameters:
    -----------
    tT_covs : array-like
        covariance matrix of temperature model
    T_params : array-like
        parameters of best fit temperature model
    tP_model_s : array-like
        time vector
    P_model_s : array-like
        modelled pressure for each scenario
    n : int
        number of samples of covariance matrix to take

    Returns:
    --------
    uncertainty_data_pressure : array-like
        modelled temperature after each scenario, taking uncertainty into account
        
    '''

    global P_model, tP_model, q_data, qt_data, dq_data

    P_model = P_model_s
    tP_model = tP_model_s

    uncertainty_model = [[], [], [], []]

    temperature_2050 = [[], [], [], []]

    [q_data, qt_data, dq_data] = [np.concatenate([q_data, [0, 0]]), np.concatenate([qt_data, [2014.6, 2050]]), np.concatenate([dq_data, [0, 0]])]

    # Solves for temperature in each uncertainty scenario
    ts = np.random.multivariate_normal(T_params, T_covs, n)
    for ti in ts:
        for i in range(4):
            P_model = P_model_s[i]
            _, T_model_uncertainty = improved_euler_solve(temperature_ode, qt_data[0], qt_data[-1], ti[2], h, ti)
            uncertainty_model[i].append(T_model_uncertainty)
            temperature_2050[i].append(T_model_uncertainty[-1])

    # Resets pressure, temperature, extraction rates back to original values
    P_model = P_model[:651]
    tP_model = tP_model[:651]
    [q_data, qt_data, dq_data] = [q_data[:-2], qt_data[:-2], dq_data[:-2]] 

    for i in range(4):
        print('Scenario', i+1, '2050 temperature 5/95% confidence interval [deg C]:', np.round(np.quantile(temperature_2050[i], [0.05, 0.95]), 3))

    return uncertainty_model

def recovery_forecast(P_covs, P_params, T_covs, T_params, R_params, tR_model_s, P_model_s, tP_model_s):
    '''
    Forecasts reservoir equilibrium temperature after four different extraction rate scenarios, taking uncertainty into account

    Parameters:
    -----------
    P_covs : array-like
        covariance matrix of pressure model
    P_params : array-like
        parameters of best fit pressure model
    T_covs : array-like
        covariance matrix of temperature model
    T_params : array-like
        parameters of best fit temperature model
    R_params : array-like
        parameters of best fit equilibrium temperature model
    tR_model_s : array-like
        time vector for equilibrium temperature
    P_model_s : array-like
        modelled pressure for each scenario
    tP_model_s
        time vector for pressure

    Returns:
    --------
    uncertainty_model : array-like
        modelled equilibrium temperature after each scenario, taking uncertainty into account
        
    '''

    global P_model, T_model, tP_model, tT_model
    
    # Generate 10 pressure/temperature models from posterior
    uncertainty_data_pressure = pressure_forecast(P_covs, P_params, 10)
    uncertainty_data_temp = temperature_forecast(T_covs, T_params, tP_model_s, P_model_s, 10)

    # Preserve temperature and pressure models
    P_model_orig = P_model
    T_model_orig = T_model

    # Vectors for uncertainty data
    uncertainty_model = [[], [], [], []]
    recovery_2050 = [[], [], [], []]

    # For each scenario, calculate 100 different equilibrium temeprature models taking uncertainty in pressure/temperature data into account
    for i in range(4):
        for j in range(10):
            for k in range(10):
                # Get pressure and temperature models
                P_model = uncertainty_data_pressure[i][j]
                T_model = uncertainty_data_temp[i][k]
                [tP_model, tT_model] = [tP_model_s, tP_model_s]
                # Calculate recovery model and add to vector of models
                R_model_uncertainty = recovery_condition(tR_model_s, R_params[0], R_params[1], R_params[2])
                uncertainty_model[i].append(R_model_uncertainty)
                # Add final equilibrium temperature value so confidence interval can be calculated later
                recovery_2050[i].append(R_model_uncertainty[-1])

    # Resets models to orignal vectors
    P_model = P_model_orig
    tP_model = tP_model[:651]
    T_model = T_model_orig
    tT_model = tT_model[:651]

    # Print equilibrium temperature confidence interval in 2050
    for i in range(4):
        print('Scenario', i+1, '2050 equilibrium temperature 5/95% confidence interval [deg C]:', np.round(np.quantile(recovery_2050[i], [0.05, 0.95]), 2))

    year_data = []
    quantile = 0

    # Find time at which lower bound of interval first exceeds 100deg C, for each scenario.
    for i in range(4):
        for j in range(len(tP_model_s)):
            for k in range(100):
                year_data.append(uncertainty_model[i][k][j])
            quantile = np.quantile(year_data, [0.05, 0.95])
            if ((quantile[0] >= 100) & (tP_model_s[j] > 1980)):
                print('Scenario', i+1, 'time at which lower bound of uncertainty interval exceeds 100deg C:', round(tP_model_s[j], 1))
                break
            year_data = []


    return uncertainty_model

def recovery_forecast_old(R_covs, R_params, tR_model_s, R_model_s, P_model_s, T_model_s):
    '''
    Forecasts reservoir equilibrium temperature after four different extraction rate scenarios, taking uncertainty into account

    Parameters:
    -----------
    tR_covs : array-like
        covariance matrix of equilibrium temperature model
    R_params : array-like
        parameters of best fit equilibrium temperature model
    tR_model_s : array-like
        time vector
    P_model_s : array-like
        modelled pressure for each scenario
    T_model_s
        modelled temperature for each scenario

    Returns:
    --------
    uncertainty_model : array-like
        modelled equilibrium temperature after each scenario, taking uncertainty into account
        
    '''
    global P_model, T_model, tP_model, tT_model

    # Adapts models to be used by recovery solver
    P_model = P_model_s
    T_model = T_model_s
    tP_model = tR_model_s
    tT_model = tR_model_s

    uncertainty_model = [[], [], [], []]

    recovery_2050 = [[], [], [], []]

    # Solves recovery condition for each uncertainty scenario
    ts = np.random.multivariate_normal(R_params, R_covs, 100)
    for ti in ts:
        for i in range(4):
            P_model = P_model_s[i]
            T_model = T_model_s[i]
            R_model_uncertainty = recovery_condition(tR_model_s, ti[0], ti[1], ti[2])
            uncertainty_model[i].append(R_model_uncertainty)
            recovery_2050[i].append(R_model_uncertainty[-1])
    
    # Resets models
    P_model = P_model[:651]
    tP_model = tP_model[:651]
    T_model = T_model[:651]
    tT_model = tT_model[:651]

    for i in range(4):
        print('Scenario', i+1, '2050 equilibrium temperature 5/95% confidence interval [deg C]:', np.round(np.quantile(recovery_2050[i], [0.05, 0.95]), 2))

    return uncertainty_model

########## Plotting functions ##########

def plot_extraction(qt_data, qt_data_total, q_data, q_data_total):
    ''' Plot the total extraction rate and the extraction from outside the rhylite 
        formation from the given data.

        Parameters:
        -----------
        qt_data : array-like
            time over which extraction data outside rhyolite formation was collected
        qt_data_total : array-like
            time over which total extraction rate data was collected
        q_data : array-like
            extraction rate data from outside rhyolite formation
        q_data_total : array-like
            total extraction rate data from Rotorua geothermal system

        Returns:
        --------
        -
    '''

    # Plot data
    f, axE = plt.subplots()
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='Waikite geyser\nlast erupts', alpha=0.7)
    plt.axvline(x=1986, color='darkorange', linestyle='dashdot', label='borehole closure\nprogram begins', alpha=0.7)
    axE.plot(qt_data_total, q_data_total/1000, 'g--', label='total extraction rate')
    axE.plot(qt_data, q_data/1000, 'g', label='extraction rate from\noutside rhyolite formation')
    
    # Title, labels, legend
    axE.set_title('Total Extraction Rate from Rotorua Geothermal System \n vs Extraction Rate outside Rhyolite Formation')
    axE.set_xlabel('time [year]')
    axE.set_ylabel('extraction rate [kt/day]')
    axE.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig('plots/extraction', bbox_inches='tight')

def plot_extraction_vs_pressure(tP_data, P_data, q_data, qt_data):
    ''' Plot the extraction rate from outside the ryholite formation against the 
        hydrostatic pressure from the given data.

        Parameters:
        -----------
        tP_data : array-like
            over which pressure data was collected
        P_data : array-like
            pressure data collected from near Whakarewarewa
        q_data : array-like
            extraction rate data from outside rhyolite formation
        qt_data : array-like
            time over which extraction data outside rhyolite formation was collected

        Returns:
        --------
        -
    '''

    # Plot data
    f, axE = plt.subplots()
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='Waikite geyser\nlast erupts', alpha=0.7)
    plt.axvline(x=1986, color='darkorange', linestyle='dashdot', label='borehole closure\nprogram begins', alpha=0.7)
    axE.plot(qt_data, q_data/1000, 'g', label='extraction rate from\noutside rhyolite formation')
    axE.plot([], [], 'b', label='hydrostatic pressure\nnear Whakarewarewa')

    # Title, labels, legend
    axE.set_title('Extraction Rate outside Rhyolite Formation\nvs Pressure near Whakarewarewa')
    axE.set_xlabel('time [year]')
    axE.set_ylabel('extraction rate [kt/day]')
    axE.legend(fontsize=8, loc='upper left')

    # Plot data on secondary axis and label
    axP = axE.twinx()
    axP.plot(tP_data, P_data/1e6, 'b')
    axP.set_ylabel('hydrostatic pressure [MPa]')
    
    plt.tight_layout()
    plt.savefig('plots/extraction_vs_presssure', bbox_inches='tight')

def plot_extraction_vs_temp(tT_data, T_data, q_data, qt_data):
    ''' Plot the extraction rate from outside the ryholite formation against
        the reservoir temperature near whakarewarewa.

        Parameters:
        -----------
        tT_data : array-like
            time over which temperature data was collected
        T_data : array-like
            temperature data collected from near Whakarewarewa
        q_data : array-like
            extraction rate data from outside rhyolite formation
        qt_data : array-like
            time over which extraction data outside rhyolite formation was collected

        Returns:
        --------
        -
    '''

    # Plot data
    f, axE = plt.subplots()
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='Waikite geyser\nlast erupts', alpha=0.7)
    plt.axvline(x=1986, color='darkorange', linestyle='dashdot', label='borehole closure\nprogram begins', alpha=0.7)
    axE.plot(qt_data, q_data/1000, 'g', label='extraction rate from\noutside rhyolite formation')
    axE.plot([], [], 'r', label='temperature near\nWhakarewarewa')

    # Title, labels, legend
    axE.set_title('Extraction Rate outside Rhyolite Formation\nvs Temperature near Whakarewarewa')
    axE.set_xlabel('time [year]')
    axE.set_ylabel('extraction rate [kt/day]')
    axE.legend(fontsize=8, loc='upper right')

    # Plot data on secondary axis and label
    axP = axE.twinx()
    axP.plot(tT_data, T_data, 'r')
    axP.set_ylabel('temperature [$^\circ$C]')
    
    plt.tight_layout()
    plt.savefig('plots/extraction_vs_temperature', bbox_inches='tight')

def plot_pressure_model(tP_data, P_data):
    ''' 
    Plots best fit pressure model and displays on screen

    Parameters:
    -----------
    tP_data : array-like
        Time values of pressure data
    P_data : array-like
        Pressure data to be calibrated to

    Returns:
    --------
    -

    '''
    # Plot data
    f, axP = plt.subplots()
    axP.plot(tP_data, P_data / 1e6, 'ko', label='data')
    axP.plot(tP_model, P_model / 1e6, 'b-', label='best fit model')

    # Title, axis, labels, legend
    axP.set_title('Calibrated Pressure Model vs Data')
    axP.set_xlabel('time [year]')
    axP.set_ylabel('pressure [MPa]')
    axP.legend(loc='lower left', fontsize=8)

    plt.savefig('plots/pressure_model', bbox_inches='tight')

def plot_pressure_misfit(tP_data, P_data):
    ''' 
    Plots misfit of best fit pressure model and displays on screen

    Parameters:
    -----------
    tP_data : array-like
        Time values of pressure data
    P_data : array-like
        Pressure data to be calibrated to

    Returns:
    --------
    -
    '''

    # Plot misfit 
    f, axPm = plt.subplots()
    P_misfit = np.interp(tP_data, tP_model, P_model) - P_data
    axPm.plot(tP_data, P_misfit, 'bx')

    # Axis, titles, labels, legend
    axPm.set_title('Misfit of Pressure Model Against Data')
    axPm.set_xlabel('time [years]')
    axPm.set_ylabel('misfit [Pa]')
    axPm.axhline(y=0, linestyle='dotted', color='k')

    plt.savefig('plots/pressure_misfit', bbox_inches='tight')

def plot_temperature_model(tT_data, T_data, tT_model, T_model):
    ''' 
    Plots best fit temperature model and displays on screen

    Parameters:
    -----------
    tT_data : array-like
        Time values of temperature data
    T_data : array-like
        Temperature data to be calibrated to
    tT_model : array-like
        Time values of temperature model
    T_model : array-like
        Temperature values of best fit model

    Returns:
    --------
    -

    '''
    # Plot data
    f, axT = plt.subplots()
    axT.plot(tT_data, T_data, 'ko', label='data')
    axT.plot(tT_model, T_model, 'r-', label='best fit model')

    # Title, labels, legend
    axT.set_title('Calibrated Temperature Model vs Data')
    axT.set_xlabel('time [year]')
    axT.set_ylabel('temperature [{}C]'.format(chr(176)))
    axT.legend(loc='lower left', fontsize=8)

    plt.savefig('plots/temperature_model', bbox_inches='tight')

def plot_temperature_misfit(tT_data, T_data, tT_model, T_model):
    ''' 
    Plots best fit temperature model and displays on screen

    Parameters:
    -----------
    tT_data : array-like
        Time values of temperature data
    T_data : array-like
        Temperature data to be calibrated to
    tT_model : array-like
        Time values of temperature model
    T_model : array-like
        Temperature values of best fit model

    Returns:
    --------
    -

    '''
    # Plot data
    f, axTm = plt.subplots()
    T_misfit = np.interp(tT_data, tT_model, T_model) - T_data
    axTm.plot(tT_data, T_misfit, 'rx')

    # Title, labels, legend
    axTm.set_title('Misfit of Temperature Model Against Data')
    axTm.set_xlabel('time [year]')
    axTm.set_ylabel('misfit [{}C]'.format(chr(176)))
    axTm.axhline(y=0, linestyle='dotted', color='k')

    plt.savefig('plots/temperature_misfit', bbox_inches='tight')

def plot_recovery_condition(tR_data, R_data, tR_model, R_model):
    ''' 
    Plots best fit recovery condition model and displays on screen

    Parameters:
    -----------
    tR_data : array-like
        Time values of equilibrium temperature data
    R_data : array-like
        Equilibrium temperature data to be calibrated to
    tR_model : array-like
        Time values of equilibrium temperature temperature model
    R_model : array-like
        Equilibrium temperature values of best fit model

    Returns:
    --------
    -

    '''
    # Plot data
    f, axRc = plt.subplots()
    plt.axhline(y=100, color='r', linestyle='dashdot', label='equilibrium temperature\nthreshold for eruption', alpha=0.8)
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='most recent eruption\nof Waikite geyser', alpha=0.8)
    axRc.plot(tR_model, R_model, 'g-', label='best fit model')

    # Title, labels, legend
    axRc.set_title('Calibrated Recovery Condition Model')
    axRc.set_xlabel('time [year]')
    axRc.set_ylabel('equilibrium temperature [$^\circ$C]')
    axRc.legend(loc='lower left', fontsize=8)

    plt.savefig('plots/recovery_condition', bbox_inches='tight')

def plot_recovery_condition_alt(tR_data, R_data, tR_model, R_model):
    ''' 
    Plots best fit recovery condition model and displays on screen

    Parameters:
    -----------
    tR_data : array-like
        Time values of equilibrium temperature data
    R_data : array-like
        Equilibrium temperature data to be calibrated to
    tR_model : array-like
        Time values of equilibrium temperature temperature model
    R_model : array-like
        Equilibrium temperature values of best fit model

    Returns:
    --------
    -

    '''
    # Plot data
    f, axRc = plt.subplots()
    plt.axhline(y=100, color='r', linestyle='dashdot', label='equilibrium temperature\nthreshold for eruption', alpha=0.8)
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='most recent eruption\nof Waikite geyser', alpha=0.8)
    axRc.plot(tR_model, R_model, 'g-', label='best fit model')

    # Title, labels, legend
    axRc.set_title('Calibrated Recovery Condition Model')
    axRc.set_xlabel('time [year]')
    axRc.set_ylabel('equilibrium temperature [$^\circ$C]')
    axRc.legend(loc='lower left', fontsize=8)

    plt.savefig('plots/recovery_condition_alt', bbox_inches='tight')

def plot_pressure_scenario_model(tP_model_s, P_model_s):
    ''' 
    Plots pressure scenario model and displays on screen

    Parameters:
    -----------
    tP_model_s : array-like
        Time vector
    P_model_s : array-like
        Data for each pressure scenario

    Returns:
    --------
    -
    '''
    
    # Plot data
    f, axPs = plt.subplots()
    axPs.plot(tP_model_s, P_model_s[0]/1e6, 'm', label='$q_{0}$ = 0')
    axPs.plot(tP_model_s, P_model_s[1]/1e6, 'g', label='$q_{0}$ = 500 t/day')
    axPs.plot(tP_model_s, P_model_s[2]/1e6, 'r', label='$q_{0}$ = 250 t/day')
    axPs.plot(tP_model_s, P_model_s[3]/1e6, 'b', label='$q_{0}$ = 1000 t/day')
    axPs.plot(tP_model[:651], P_model[:651]/1e6, 'k', label='model')
    axPs.plot(tP_data, P_data/1e6, 'bo', label='data')

    # Axis, labels, legend
    axPs.set_title('RGS LPM: what-if scenarios (pressure)')
    axPs.set_xlabel('time [year]')
    axPs.set_ylabel('pressure [MPa]')
    axPs.legend(loc = 'lower left', fontsize = 8)

    plt.savefig('plots/pressure_scenario_model', bbox_inches='tight')

def plot_temperature_scenario_model(tT_model, T_model, tT_model_s, T_model_s):
    ''' 
    Plots pressure scenario model and displays on screen

    Parameters:
    -----------
    tT_model : array-like
        Temperature model time vector
    T_model : array-like
        Modelled temperature
    tT_model_s : 
        Time vector for each temperature scenario
    T_model_s :
        Modelled temperature for each temperature scenario

    Returns:
    --------
    -
    '''
    
    # Plot temperature scenario model
    f, axTs = plt.subplots()
    axTs.plot(tT_model_s, T_model_s[0], 'm', label='$q_{0}$ = 0')
    axTs.plot(tT_model_s, T_model_s[1], 'g', label='$q_{0}$ = 500 t/day')
    axTs.plot(tT_model_s, T_model_s[2], 'r', label='$q_{0}$ = 250 t/day')
    axTs.plot(tT_model_s, T_model_s[3], 'b', label='$q_{0}$ = 1000 t/day')
    axTs.plot(tT_model, T_model, 'k', label='model')
    axTs.plot(tT_data, T_data, 'ro', label='data')

    # Axis, labels, legend
    axTs.set_title('RGS LPM: what-if scenarios (temperature)')
    axTs.set_xlabel('time [year]')
    axTs.set_ylabel('temperature [$^\circ$C]')
    axTs.legend(loc = 'lower left', fontsize = 8)
    
    plt.savefig('plots/temperature_scenario_model', bbox_inches='tight')

def plot_recovery_scenario_model(tR_model, R_model, tR_model_s, R_model_s):
    ''' 
    Plots pressure scenario model and displays on screen

    Parameters:
    -----------
    tT_model : array-like
        Temperature model time vector
    T_model : array-like
        Modelled temperature
    tT_model_s : 
        Time vector for each temperature scenario
    T_model_s :
        Modelled temperature for each temperature scenario

    Returns:
    --------
    -
    '''
    
    # Plot temperature scenario model
    f, axRs = plt.subplots()
    plt.axhline(y=100, color='r', linestyle='dashdot', label='equilibrium temperature\nthreshold for eruption', alpha=0.7)
    plt.axvline(x=1967, color='k', linestyle='dashdot', label='most recent eruption\nof Waikite geyser', alpha=0.7)
    axRs.plot(tR_model_s, R_model_s[0], 'm', label='$q_{0}$ = 0')
    axRs.plot(tR_model_s, R_model_s[1], 'g', label='$q_{0}$ = 500 t/day')
    axRs.plot(tR_model_s, R_model_s[2], 'r', label='$q_{0}$ = 250 t/day')
    axRs.plot(tR_model_s, R_model_s[3], 'b', label='$q_{0}$ = 1000 t/day')
    axRs.plot(tR_model, R_model, 'k', label='model')

    # Axis, labels, legend
    axRs.set_title('Waikite geyser: what-if scenarios')
    axRs.set_xlabel('time [year]')
    axRs.set_ylabel('equilibrium temperature [$^\circ$C]')
    axRs.legend(loc = 'lower right', fontsize = 8)
    
    plt.savefig('plots/recovery_scenario_model', bbox_inches='tight')

def plot_pressure_forecast(uncertainty_data, tP_model, tP_model_s):
    ''' 
    Plots pressure forecast, taking uncertainty into account, and displays on screen

    Parameters:
    -----------
    uncertainty_data : array-like
        Uncertainty data for each pressure scenario
    tP_model : array-like
        Time vector 
    tP_model_s : 
        Time vector for each scenario

    Returns:
    --------
    -
    '''
    
    fig,axPu = plt.subplots(2,2)

    # Plots forecasted pressure, for each scenario, for each set of parameters
    for i in range(100):
        axPu[0][0].plot(tP_model_s[651:], uncertainty_data[0][i][651:]/1e6, 'm-', alpha=0.2, lw=0.5)
        axPu[0][1].plot(tP_model_s[651:], uncertainty_data[1][i][651:]/1e6, 'g-', alpha=0.2, lw=0.5)
        axPu[1][0].plot(tP_model_s[651:], uncertainty_data[2][i][651:]/1e6, 'r-', alpha=0.2, lw=0.5)
        axPu[1][1].plot(tP_model_s[651:], uncertainty_data[3][i][651:]/1e6, 'b-', alpha=0.2, lw=0.5)
        for j in range(2):
            for k in range(2):
                axPu[j][k].plot(tP_model_s[:651], uncertainty_data[0][i][:651]/1e6, 'k-', alpha=0.2, lw=0.5)

    axPu[0][0].plot([], [], 'm-', label='$q_{0}$ = 0')
    axPu[0][1].plot([], [], 'g-', label='$q_{0}$ = 500 t/day')
    axPu[1][0].plot([], [], 'r-', label='$q_{0}$ = 250 t/day')
    axPu[1][1].plot([], [], 'b-', label='$q_{0}$ = 1000 t/day')

    # Plot data and create legend
    for i in range(2):
        for j in range(2):
            axPu[i][j].errorbar(tP_data, P_data/1e6, yerr=800/1e6, fmt='bo', ms=2)
            axPu[i][j].legend(fontsize=8, loc='lower right')
        
    fig.text(0.5, 0.93, 'RGS LPM: scenario forecasts (pressure)', ha='center', fontsize=12)
    fig.text(0.5, 0.02, 'time [year]', ha='center')
    fig.text(0.01, 0.5, 'pressure [MPa]', va='center', rotation='vertical')

    for ax in axPu.flat:
        ax.label_outer()

    plt.savefig('plots/pressure_forecast', bbox_inches='tight')

def plot_temperature_forecast(uncertainty_data, tT_model, tT_model_s):
    ''' 
    Plots temperature forecast, taking uncertainty into account, and displays on screen

    Parameters:
    -----------
    uncertainty_data : array-like
        Uncertainty data for each temperature scenario
    tT_model : array-like
        Time vector 
    tT_model_s : 
        Time vector for each scenario

    Returns:
    --------
    -
    '''
    
    fig,axTu = plt.subplots(2,2)

    # Plots forecasted pressure, for each scenario, for each set of parameters
    for i in range(100):
        axTu[0][0].plot(tT_model_s[651:], uncertainty_data[0][i][651:], 'm-', alpha=0.2, lw=0.5)
        axTu[0][1].plot(tT_model_s[651:], uncertainty_data[1][i][651:], 'g-', alpha=0.2, lw=0.5)
        axTu[1][0].plot(tT_model_s[651:], uncertainty_data[2][i][651:], 'r-', alpha=0.2, lw=0.5)
        axTu[1][1].plot(tT_model_s[651:], uncertainty_data[3][i][651:], 'b-', alpha=0.2, lw=0.5)
        for j in range(2):
            for k in range(2):
                axTu[j][k].plot(tT_model_s[:651], uncertainty_data[0][i][:651], 'k-', alpha=0.2, lw=0.5)

    # Title, labels, legend
    fig.text(0.5, 0.93, 'RGS LPM: scenario forecasts (temperature)', ha='center', fontsize=12)
    fig.text(0.5, 0.02, 'time [year]', ha='center')
    fig.text(0.01, 0.5, 'temperature [$^\circ$C]', va='center', rotation='vertical')

    axTu[0][0].plot([], [], 'm-', label='$q_{0}$ = 0')
    axTu[0][1].plot([], [], 'g-', label='$q_{0}$ = 500 t/day')
    axTu[1][0].plot([], [], 'r-', label='$q_{0}$ = 250 t/day')
    axTu[1][1].plot([], [], 'b-', label='$q_{0}$ = 1000 t/day')

    # Plot data with error bars
    for i in range(2):
        for j in range(2):
            axTu[i][j].errorbar(tT_data, T_data, yerr=1, fmt='ro', ms=2)
            axTu[i][j].legend(fontsize=8, loc='lower right')
    
    for ax in axTu.flat:
        ax.label_outer()

    plt.savefig('plots/temperature_forecast', bbox_inches='tight')

def plot_recovery_forecast(uncertainty_data, tR_model, tR_model_s):
    ''' 
    Plots geyser recovery forecast, taking uncertainty into account, and displays on screen

    Parameters:
    -----------
    uncertainty_data : array-like
        Uncertainty data for each temperature scenario
    tT_model : array-like
        Time vector 
    tT_model_s : 
        Time vector for each scenario

    Returns:
    --------
    -
    '''
    
    fig,axRu = plt.subplots(2,2)

    # define colours of lines to be used on each plot
    styles = [['m-', 'g-'], ['r-', 'b-']]

    # Plot eruption threshold
    for i in range(2):
        for j in range(2):
            axRu[i][j].axhline(y=100, color='r', linestyle='dashdot', alpha=0.7)
            axRu[i][j].axvline(x=1967, color='k', linestyle='dashdot', alpha=0.7)

    # Plots forecasted pressure, for each scenario, for each set of parameters
    for i in range(100):
        axRu[0][0].plot(tR_model_s[651:], uncertainty_data[0][i][651:], 'm-', alpha=0.2, lw=0.5)
        axRu[0][1].plot(tR_model_s[651:], uncertainty_data[1][i][651:], 'g-', alpha=0.2, lw=0.5)
        axRu[1][0].plot(tR_model_s[651:], uncertainty_data[2][i][651:], 'r-', alpha=0.2, lw=0.5)
        axRu[1][1].plot(tR_model_s[651:], uncertainty_data[3][i][651:], 'b-', alpha=0.2, lw=0.5)
        for j in range(2):
            for k in range(2):
                axRu[j][k].plot(tR_model_s[:651], uncertainty_data[0][i][:651], 'k-', alpha=0.2, lw=0.5)


    # Titles, label, legend
    axRu[0][0].plot([], [], 'm-', label='$q_{0}$ = 0')
    axRu[0][1].plot([], [], 'g-', label='$q_{0}$ = 500 t/day')
    axRu[1][0].plot([], [], 'r-', label='$q_{0}$ = 250 t/day')
    axRu[1][1].plot([], [], 'b-', label='$q_{0}$ = 1000 t/day')

    for i in range(2):
        for j in range(2):
            axRu[i][j].legend(fontsize=8, loc='lower right')

    # Title/labels
    fig.text(0.5, 0.93, 'Waikite geyser: scenario forecasts', ha='center', fontsize=12)
    fig.text(0.5, 0.02, 'time [year]', ha='center')
    fig.text(0.01, 0.5, 'equilibrium temperature [$^\circ$C]', va='center', rotation='vertical')
        
    for ax in axRu.flat:
        ax.label_outer()

    #plt.tight_layout()
    
    plt.savefig('plots/recovery_forecast', bbox_inches='tight')
