#!/usr/bin/env sh
# This is a script to run the tests 3 times, once to write the coverage of each specific module.

pytest mesa_ret/testsret --cov=mesa_ret/mesa_ret --cov-report=html:htmlcov/mesa_ret-coverage
pytest mesa_ret/testsret --cov=mesa_ret/retgen --cov-report=html:htmlcov/retgen-coverage
pytest mesa_ret/testsret --cov=mesa_ret/retplay --cov-report=html:htmlcov/retplay-coverage

read -p "Test script has finished. Press any key to continue"
