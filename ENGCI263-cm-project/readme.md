# ENGSCI263 Geyser Recovery in Rotorua
ENGSCI263 project 1 on geyser recovery in Rotorua.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Use](#use)
* [Contact](#contact)

## General info
This project aims to model the pressure and temperature of the Rotorua geothermal field to speculate on geyser recovery in order to make a recommendation for action by the Bay of Plenty Regional Council. It includes figures of the calibrated models and uncertainty plots along with unit tests and benchmarking.

## Technologies
Project is created with:
* Python 3.7

## Setup
To generate all the figures we used for this project, run the script ``main.py`` (followed by ``outdated_model.py`` if you are interested in viewing the older versions of these figures, which some of us are using). This will save all relevant figures to the ``plots`` folder. 

## Features
* ``main.py`` is the script which runs all the functions in the other python scripts to display the relevant figures.
* ``improved_euler_functions.py`` contains functions for applying the improved euler method.
* ``calibration_and_forecasts.py`` contains functions to read data and create plots.
* ``unit_tests_benchmarks.py`` contains unit test functions and benchmarking functions for temperature, pressure, and ode models.
* ``outdated_model.py`` contains previous outdated models of pressure and temperature.

## Use
* ``main.py`` is used to run all the unit tests and benchmarks and then create the plots.
* ``improved_euler_functions.py`` is used for numerical analysis to create our models.
* ``calibration_and_forecasts.py`` is used to implement the numerical analysis of our ode models, calibrate our models to the observed data, and contains the code for plotting the model and data. Plots include scenario prediction, uncertainty, and recovery forecasts.
* ``unit_tests_benchmarks.py`` contains unit test functions and benchmarking functions for temperature, pressure, and ode models.
* ``outdated_model.py`` is used to create plots of the previous outdated models which can be compared to the current ones.

## Contact
Created by Group Alex
