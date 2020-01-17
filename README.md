# open-eureqa

The basic idea of this project is to re-create at least *some* functionalities of the commercial software Eureqa Formulize, building on existing Python packages that provide utilities for Genetic Programming (gplearn) and multi-objective evolutionary optimization à la NSGA-II (inspyred).

## Required packages
The project is dependant on at least **gplearn** (https://gplearn.readthedocs.io/en/stable/intro.html) and **inspyred** (https://pythonhosted.org/inspyred/)-

## Eureqa functionalities

There are many functions of Eureqa Formulize that would be cool to re-create. Unfortunately, several are pretty complex.

1. Pareto front complexity-fitting (this should be the easiest);
2. Competitive co-evolution between 8 points of the training set and the current population of candidate solutions;
3. Simplification of the equations during the run (maybe through specific operators?);
4. Caching of some parts of the functions, to speed up evaluations (probably impractical);
