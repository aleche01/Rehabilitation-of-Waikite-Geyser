import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from improved_euler_functions import *

def pressure_ode_custom_inputs(t, P, ap, bp, cp, P0, q, dq):
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
        q   : float
            Extraction rate (assume constant).
        dq  : float
            Rate of change of extraction rate (assume constant).

        Returns:
        --------
        dPdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
    return -ap * q - bp * (P - P0) - cp * dq


def temperature_ode_custom_inputs(t, T, aT, bT, Tc, T0, ap, bp, P0, P, tP=None):
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
        P  : float/array
            Pressure, constant value or array.
        tP : float/array
            Time for pressure, default None if pressure is constant.

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
    if tP is None:
        Pt = P
    else:
        Pt = np.interp(t, tP, P)
    return -aT * (bp / ap) * (Pt - P0) * (Tc - T) - bT * (T - T0)


def test_pressure_ode():
    """
	Test if function pressure_ode is working properly by comparing it with a known result.

	Remember to consider any edge cases that may be relevant.
    
    Note: pressure_ode uses global variables in a separate script, so pressure_ode_custom_inputs
    is a replica that takes paramaters rather than global variables.
	"""
    #general test case
    dpdt = pressure_ode_custom_inputs(1, 2, 3, 4, 5, 6, 7, 8)
    assert(dpdt == -45)

    #zero term
    dpdt = pressure_ode_custom_inputs(1, 2, 0, 4, 5, 6, 7, 8)
    assert(dpdt == -24)
    
    print("pressure_ode PASSED ALL TESTS")


def test_temperature_ode():
    """
	Test if function temperature_ode is working properly by comparing it with a known result.

	Remember to consider any edge cases that may be relevant.
    
    Note: temperature_ode uses global variables in a separate script, so temperature_ode_custom_inputs
    is a replica that takes paramaters rather than global variables.
	"""
    #general test case
    dTdt = temperature_ode_custom_inputs(1, 2, 3, 4, 5, 6, 7, 7, 9, 10)
    assert(dTdt == 7)

    #zero term 
    dTdt = temperature_ode_custom_inputs(1, 2, 0, 4, 5, 6, 7, 7, 9, 10)
    assert(dTdt == 16) 


    print("temperature_ode PASSED ALL TESTS")


def test_improved_euler_step():
    """
	Test if function improved_euler_step is working properly by comparing it with a known result.

	Tests the postive starting point, zero gradient, and negative starting point. 
	"""
    #positive starting point
    pars = [3, 4, 5, 6, 7, 7, 9, 10]
    Y1 = improved_euler_step(temperature_ode_custom_inputs, 0, 10.0, 0.1, pars)
    assert(Y1 == 9.905)

    #zero gradient case
    pars = [0, 0, 5, 6, 7, 7, 9, 10]
    Y1 = improved_euler_step(temperature_ode_custom_inputs, 0, 10.0, 0.1, pars)
    assert(Y1 == 10)

    #negative starting point
    pars = [1, 4, 4, -6, 7, 2, 3, 10]
    Y1 = improved_euler_step(temperature_ode_custom_inputs, 0, -10, 10, pars)
    assert(Y1 == 1070)

    print("improved_euler_step PASSED ALL TESTS")


def test_improved_euler_solve():
    """
	Test if function improved_euler_solve is working properly by comparing it with a known result.

	Tests the postive starting point, zero gradient, negative starting point, and the creation of the t array. 
	"""
    tol = 1.e-6

    #positive starting point
    pars = [3, 4, 5, 6, 7, 7, 9, 10]
    t, Y = improved_euler_solve(temperature_ode_custom_inputs, 0, 0.2, 10.0, 0.1, pars)
    t_soln = [0,0.1,0.2]
    Y_soln = [10,9.905,9.819025]
    assert norm(t - t_soln) < tol
    assert norm(Y - Y_soln) < tol

    #zero gradient case
    pars = [0, 0, 5, 6, 7, 7, 9, 10]
    t, Y = improved_euler_solve(temperature_ode_custom_inputs, 0, 0.2, 10.0, 0.1, pars)
    t_soln = [0,0.1,0.2]
    Y_soln = [10,10,10]
    assert norm(t - t_soln) < tol
    assert norm(Y - Y_soln) < tol

    #negative starting point
    pars = [1, 4, 4, -6, 7, 2, 3, 10]
    t, Y = improved_euler_solve(temperature_ode_custom_inputs, 0, 20, -10.0, 10, pars)
    t_soln = [0,10,20]
    Y_soln = [-10,1070,196550]
    assert norm(t - t_soln) < tol
    assert norm(Y - Y_soln) < tol

    #testing step size and length/stepsize incompatibility
    pars = [3, 4, 5, 6, 7, 7, 9, 10]
    t, Y = improved_euler_solve(temperature_ode_custom_inputs, 0, 0.2, 10.0, 0.09, pars)
    t_soln = [0,0.09,0.18]
    assert norm(t - t_soln) < tol

    print("improved_euler_solve PASSED ALL TESTS")


def benchmarks():
    ''' Plots benchmarks for pressure and temperature ODEs, along with
        error analysis and timestep convergence.
        
        Notes:
        ------
        Analytical solution for pressure requires that extraction rate
        be constant, while analytical solution for temperature is dependent
        on pressure but requires conduction to be 0.
    '''
    # set parameters
    ap = 1.
    bp = 0.5
    cp = 0.
    q = 1.
    dq = 0.
    P0 = 3.
    aT = 1.
    bT = 0.
    Tc = 3.
    T0 = 5.

    # set time range and range of values for h
    t0 = 0
    t1 = 8
    h = np.logspace(0.25, -0.5)

    # set values for h to visibly compare
    good_h = 0.2
    bad_h = 1.5

    # create range of values for h, and initialise arrays
    final_model_pressures = []
    final_analytical_pressures = []
    final_model_temperatures = []
    final_analytical_temperatures = []
    
    # calculate values for timestep convergence analysis
    for step_size in h:
        time, pressure = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, P0, step_size, [ap, bp, cp, P0, q, dq])
        final_model_pressures.append(pressure[-1])
        final_analytical_pressures.append(P0 - ap * q / bp * (1 - np.exp(-bp * time[-1])))

        time, temperature = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, T0, step_size, [aT, bT, Tc, T0, ap, bp, P0, pressure, time])
        final_model_temperatures.append(temperature[-1])
        final_analytical_temperatures.append(Tc + (T0 - Tc) * np.exp(-aT * q / bp * (np.exp(-bp * time[-1] + bp * time[1] - 1))))

    # calculate values for benchmark and error analysis
    t_good_model, P_good_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, P0, good_h, [ap, bp, cp, P0, q, dq])
    t_bad_model, P_bad_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, P0, bad_h, [ap, bp, cp, P0, q, dq])
    P_analytical = P0 - ap * q / bp * (1 - np.exp(-bp * t_good_model))
    P_error = []

    t_good_model, T_good_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, T0, good_h, [aT, bT, Tc, T0, ap, bp, P0, P_good_model, t_good_model])
    t_bad_model, T_bad_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, T0, bad_h, [aT, bT, Tc, T0, ap, bp, P0, P_bad_model, t_bad_model])
    T_analytical = Tc + (T0 - Tc) * np.exp(-aT * q / bp * (np.exp(-bp * t_good_model) + bp * t_good_model - 1))
    T_error = []

    for i in range(len(t_good_model)):
        P_error.append(abs(P_good_model[i] - P_analytical[i]) * 1000)
        T_error.append(abs(T_good_model[i] - T_analytical[i]) * 1000)
    

    ### Pressure Benchmark ###
    # plot
    fP, axPb = plt.subplots(ncols=3)
    fP.set_size_inches(12, 9)

    # benchmark
    axPb[0].hlines(P0 - ap * q / bp, t0, t1, colors='g', linestyles='dashed', label='steady-state solution')
    axPb[0].plot(t_good_model, P_good_model, 'b*', label='model (step size = {:.1f})'.format(good_h))
    axPb[0].plot(t_bad_model, P_bad_model, 'r*', label='model (step size = {:.1f})'.format(bad_h))
    axPb[0].plot(t_good_model, P_analytical, 'g-', label='analytical solution')
    axPb[0].legend()
    axPb[0].set_title('Benchmark of Pressure ODE\nap = {:.1f}, bp = {:.1f}, cp = {:.1f}, q = {:.1f}'.format(ap, bp, cp, q))
    axPb[0].set_xlabel('t')
    axPb[0].set_ylabel('P(t)')

    # error analysis
    axPb[1].plot(t_good_model, P_error, 'r--')
    axPb[1].set_title('Error Analysis')
    axPb[1].set_xlabel('t')
    axPb[1].set_ylabel('model error against benchmark [P x 1.e3]')
    axPb[1].yaxis.tick_right()

    # timestep convergence analysis
    axPb[2].plot(1 / h, final_model_pressures, 'ko', markersize=4)
    axPb[2].set_title('Timestep Convergence Analysis')
    axPb[2].set_xlabel('1 / step size')
    axPb[2].set_ylabel('P(t = {:.1f})'.format(t1))
    axPb[2].yaxis.set_label_position("right")
    axPb[2].yaxis.tick_right()

    # save fig
    plt.savefig('plots/pressure_benchmark', bbox_inches='tight')


    ### Temperature Benchmark ###
    # plot
    f, axTb = plt.subplots(ncols=3)
    f.set_size_inches(12, 9)

    # benchmark
    axTb[0].hlines(Tc, t0, t1, colors='g', linestyles='dashed', label='steady-state solution')
    axTb[0].plot(t_good_model, T_good_model, 'b*', label='model (step size = {:.1f})'.format(good_h))
    axTb[0].plot(t_bad_model, T_bad_model, 'r*', label='model (step size = {:.1f})'.format(bad_h))
    axTb[0].plot(t_good_model, T_analytical, 'g-', label='analytical solution')
    axTb[0].legend()
    axTb[0].set_title('Benchmark of Temperature ODE\naT = {:.1f}, bT = {:.1f}'.format(aT, bT))
    axTb[0].set_xlabel('t')
    axTb[0].set_ylabel('T(t)')

    # error analysis
    axTb[1].plot(t_good_model, T_error, 'r--')
    axTb[1].set_title('Error Analysis')
    axTb[1].set_xlabel('t')
    axTb[1].set_ylabel('model error against benchmark [T x 1.e3]')
    axTb[1].yaxis.tick_right()

    # timestep convergence analysis
    axTb[2].plot(1 / h, final_model_temperatures, 'ko', markersize=4)
    axTb[2].set_title('Timestep Convergence Analysis\n')
    axTb[2].set_xlabel('1 / step size')
    axTb[2].set_ylabel('T(t = {:.1f})'.format(t1))
    axTb[2].yaxis.set_label_position("right")
    axTb[2].yaxis.tick_right()

    # save fig
    plt.savefig('plots/temperature_benchmark', bbox_inches='tight')


def pressure_gold_secret(save=False):
    ''' Plots David's Gold Secret for pressure ODE.
        
        Notes:
        ------
        Changes parameter values to extremes, then runs models.
    '''
    # randomly selected parameters
    ap = 1
    bp = 1
    cp = 1
    q = 1
    dq = 1
    P0 = 10
    
    # set initial pressure and time range
    Pt0 = 1
    t0 = 0
    t1 = 10

    # step size
    h = 0.5

    f, axPgs = plt.subplots(2, 2)
    
    # setting aq to be very large
    t_model, ap_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, Pt0, h, [ap * 100, bp, cp, P0, q, dq])
    axPgs[0, 0].plot(t_model, ap_model, 'r--', label = 'ap >> bp, cp, q')
    axPgs[0, 0].legend()
    axPgs[0, 0].set_title('Pressure Model Solutions\nat Extreme Values')
    axPgs[0, 0].set_ylabel('P(t)')

    # setting all except bp to equal 0
    bp_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, Pt0, h, [ap * 0, bp, cp * 0, P0, q * 0, dq])[1]
    axPgs[0, 1].plot(t_model, bp_model, 'b--', label = 'ap, cp, q = 0')
    axPgs[0, 1].legend()

    # setting cp to be very large
    cp_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, Pt0, h, [ap, bp, cp * 100, P0, q, dq])[1]
    axPgs[1, 1].plot(t_model, cp_model, 'g--', label = 'cp >> ap, bp, q')
    axPgs[1, 1].legend()
    axPgs[1, 1].set_xlabel('t')

    # setting ambient pressure much larger than current
    q_model = improved_euler_solve(pressure_ode_custom_inputs, t0, t1, Pt0, h, [ap, bp, cp, P0 * 50, q, dq])[1]
    axPgs[1, 0].plot(t_model, q_model, 'y--', label = 'P0 >> P')
    axPgs[1, 0].legend()
    axPgs[1, 0].set_xlabel('t')
    axPgs[1, 0].set_ylabel('P(t)')

    plt.savefig('plots/pressure_gold_secret', bbox_inches='tight')
    

def temperature_gold_secret(save=False):
    ''' Plots David's Gold Secret for temperature ODE.

        Notes:
        ------
        Changes parameter values to extremes, then runs models.
        Assumes pressure to be constant.
    '''
    # randomly selected parameters
    aT = 1.
    bT = 1.
    Tc = 5.
    T0 = 15.
    ap = 1.
    bp = 1.
    P0 = 10.
    P = 9.
    
    # set initial temperature and time range
    Tt0 = 10.
    t0 = 0.
    t1 = 10.

    # step size
    h = 0.1

    f, axTgs = plt.subplots(2, 2)
    
    # setting aT to be very large
    t_model, aT_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, Tt0, h, [aT, bT * 0, Tc, T0, ap, bp, P0, P])
    axTgs[0, 0].plot(t_model, aT_model, 'r--', label = 'bT = 0')
    axTgs[0, 0].legend()
    axTgs[0, 0].set_title('Temperature Model Solutions\nat Extreme Values')
    axTgs[0, 0].set_ylabel('T(t)')

    # setting aT to equal 0
    bT_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, Tt0, h, [aT * 0, bT, Tc, T0, ap, bp, P0, P])[1]
    axTgs[0, 1].plot(t_model, bT_model, 'b--', label = 'aT = 0')
    axTgs[0, 1].legend()

    # setting P to be very small
    diffP_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, Tt0, h, [aT, bT, Tc, T0, ap, bp, P0 * 2, P])[1]
    axTgs[1, 1].plot(t_model, diffP_model, 'g--', label = 'P << P0')
    axTgs[1, 1].legend()
    axTgs[1, 1].set_xlabel('t')

    # setting ambient temp much greater than current
    T0_model = improved_euler_solve(temperature_ode_custom_inputs, t0, t1, Tt0, h, [aT, bT, Tc, T0 * 50, ap, bp, P0, P])[1]
    axTgs[1, 0].plot(t_model, T0_model, 'y--', label = 'T0 >> T')
    axTgs[1, 0].legend()
    axTgs[1, 0].set_xlabel('t')
    axTgs[1, 0].set_ylabel('T(t)')

    plt.savefig('plots/temperature_gold_secret', bbox_inches='tight')


def run_tests():
    """
    Runs all unit tests for derivative functions and iterative method functions.
    """
    test_pressure_ode()
    test_temperature_ode()
    test_improved_euler_step()
    test_improved_euler_solve()


def run_benchmarks(save=False):
    """
    Plots all benchmarks for pressure and temperature ODES.
    """
    benchmarks()
    pressure_gold_secret()
    temperature_gold_secret()
