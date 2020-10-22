from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import numpy as np
from scipy.optimize import curve_fit
from calibration_and_forecasts import *
from unit_tests_benchmarks import run_tests, run_benchmarks


def main():
    
    ########## Model generation ##########
    
    # Data generation
    tP_data, P_data, tT_data, T_data, qt_data, qt_data_total, q_data, q_data_total, dq_data, tR_data, R_data = generate_data()
    
    # Unit tests and benchmarks
    run_tests()
    run_benchmarks()

    # Model calibration
    P_params, P_covs, tP_model, P_model = calibrate_pressure_model(tP_data, P_data)
    T_params, T_covs, tT_model, T_model = calibrate_temperature_model(tT_data, T_data)
    R_params, R_covs, tR_model, R_model = calibrate_recovery_condition(tR_data, R_data)
    # Calibrate a second recovery model to illustrate that there are an infinity of potential models.
    R_params_alt, R_covs_alt, tR_model_alt, R_model_alt = calibrate_recovery_condition_alt(tR_data, R_data)

    # Scenario models
    tP_model_s, P_model_s = pressure_scenario_model()
    tT_model_s, T_model_s = temperature_scenario_model(tP_model_s, P_model_s)
    tR_model_s, R_model_s = recovery_scenario_model(tP_model_s, P_model_s, T_model_s, R_params)
    
    # Forecasts
    uncertainty_data_pressure = pressure_forecast(P_covs, P_params, 100)
    uncertainty_data_temp = temperature_forecast(T_covs, T_params, tP_model_s, P_model_s, 100)
    uncertainty_data_recovery = recovery_forecast(P_covs, P_params, T_covs, T_params, R_params, tR_model_s, P_model_s, tP_model_s)

    ########## Plotting ##########
    
    # Plot data 
    plot_extraction(qt_data, qt_data_total, q_data, q_data_total)
    plot_extraction_vs_pressure(tP_data, P_data, q_data, qt_data)
    plot_extraction_vs_temp(tT_data, T_data, q_data, qt_data)

    # Model calibration/misfit
    plot_pressure_model(tP_data, P_data)
    plot_pressure_misfit(tP_data, P_data)
    plot_temperature_model(tT_data, T_data, tT_model, T_model)
    plot_temperature_misfit(tT_data, T_data, tT_model, T_model)
    plot_recovery_condition(tR_data, R_data, tR_model, R_model)

    plot_recovery_condition_alt(tR_data, R_data, tR_model_alt, R_model_alt)
    
    # Scenario models and forecasts
    plot_pressure_scenario_model(tP_model_s, P_model_s)
    plot_temperature_scenario_model(tT_model, T_model, tT_model_s, T_model_s)
    plot_recovery_scenario_model(tR_model, R_model, tR_model_s, R_model_s)
    plot_pressure_forecast(uncertainty_data_pressure, tP_model, tP_model_s)
    plot_temperature_forecast(uncertainty_data_temp, tT_model, tT_model_s)
    plot_recovery_forecast(uncertainty_data_recovery, tR_model, tR_model_s)

    
if __name__ == "__main__":
    main()