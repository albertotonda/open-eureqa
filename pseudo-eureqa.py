"""
Script that tries to exploit gplearn's individual structure and inspyred's NSGA-II implementation to recreate some of the original features
of the commercial software Eureqa Formulize

by Alberto Tonda, 2018 <alberto.tonda@gmail.com>
"""

# commonly installed packages
import copy
import numpy as np
import random
import sys

# necessary-but-not-so-commonly-installed packages
import inspyred

from gplearn._program import _Program # base class for the GP-trees; TODO I might extend it to add an internal method that evaluates size, as defined by Eureqa-like stuff? 
from gplearn import functions
from gplearn import fitness

# useful, but maybe not strictly necessary packages
from pandas import read_csv

# local packages

"""
function that initializes the population, creating random individuals
"""
def pse_generator(random, args) :
	
	individual = _Program( 
			args["function_set"],
			args["arities"],
			args["init_depth"],
			args["init_method"],
			args["n_features"],
			args["const_range"],
			args["metric"],
			args["p_point_replace"],
			args["parsimony_coefficient"],
			args["random_state"],
			)
	individual.build_program(args["random_state"])
	while not individual.validate_program() : individual.build_program(args["random_state"])
	
	return individual

# function that evaluates an individual
"""
'evaluator' evaluates a list of individuals. Since the problem is multi-objective, both complexity and error are evaluated.
"""
def pse_evaluator(candidates, args) :
	
	target_variable_data = args["target_variable_data"]
	other_variables_data = args["other_variables_data"]
	sample_weight = args["sample_weight"]
	
	fitness_values = []

	for c in candidates :
		
		# TODO modify complexity in an 'eureqa-like' way
		complexity = len(c.__str__())
		error = c.raw_fitness(other_variables_data, target_variable_data, sample_weight)

		print("Individual: \"%s\", complexity: %d, error: %.4f" % (c, complexity, error))
		
		fitness_values.append( inspyred.ec.emo.Pareto( [error, complexity] ) )

	return fitness_values

# global 'variation' function that just calls _Program methods with different probabilities
"""
'variator' creates new individuals calling the functions of class '_Program'
"""
@inspyred.ec.variators.crossover # decorator that requires the variation function to accept two 'parent' individuals
def pse_variator(random, parent1, parent2, args) :
	
	children = []
	
	#parent1 = copy.deepcopy(parent1)
	#parent2 = copy.deepcopy(parent2)
	
	program1 = None
	program2 = None
	
	# TODO some default hard-coded probabilities...to be changed
	p_crossover = 0.8
	p_subtree_mutation = 0.01
	p_hoist_mutation = 0.01
	p_point_mutation = 0.01

	# TODO 	understand better how the different functions work; apparently, they do not
	#	modify the individual, but return a new 'program' that is later used to initialize an individual
	if random.random() < p_crossover :
		program1, removed, remains = parent1.crossover(parent2.program, args["random_state"])
		program2, removed, remains = parent2.crossover(parent1.program, args["random_state"])
	
	elif random.random() < p_crossover + p_subtree_mutation :
		program1, removed, _ = parent1.subtree_mutation(args["random_state"])
		program2, removed, _ = parent2.subtree_mutation(args["random_state"])
	
	elif random.random() < p_crossover + p_subtree_mutation + p_hoist_mutation :
		program1, removed = parent1.hoist_mutation(args["random_state"])
		program2, removed = parent2.hoist_mutation(args["random_state"])

	# TODO: point_mutation has an issue with 'arity' (that should be a list or something, it's an int instead)
	#if random.random() < p_point_mutation :
	#	program1, mutated = parent1.point_mutation(args["random_state"])
	#	program2, mutated = parent2.point_mutation(args["random_state"])

	if program1 != None and program2 != None :

		child1 = _Program(
			args["function_set"],
			args["arities"],
			args["init_depth"],
			args["init_method"],
			args["n_features"],
			args["const_range"],
			args["metric"],
			args["p_point_replace"],
			args["parsimony_coefficient"],
			args["random_state"],
			program=program1
			)

		child2 = _Program(
			args["function_set"],
			args["arities"],
			args["init_depth"],
			args["init_method"],
			args["n_features"],
			args["const_range"],
			args["metric"],
			args["p_point_replace"],
			args["parsimony_coefficient"],
			args["random_state"],
			program=program2
			)
		
		#print("child1=", child1)
		#print("child2=", child2)
		#if child1.validate_program() : children.append( copy.deepcopy(child1) )
		#if child2.validate_program() : children.append( copy.deepcopy(child2) )
		if child1.validate_program() : children.append( child1 )
		if child2.validate_program() : children.append( child2 )
	
	return children
	
"""
'observer' function that is called at the end of each generation
"""
def pse_observer(population, num_generations, num_evaluations, args) :
	
	# find individuals with best fitting and lowest complexity
	best_f = min(population, key=lambda x : x.fitness[0]) 
	best_c = min(population, key=lambda x : x.fitness[1]) 
	
	# TODO: find individual near the 'knee'
	
	# print some stats to screen
	print("\nGeneration %d, evaluations=%d" % (num_generations, num_evaluations))
	print("\t- Best fitting: \"%s\", complexity: %d, error: %.4f" % (best_f.candidate, best_f.fitness[1], best_f.fitness[0]))
	print("\t- Best complexity: \"%s\", complexity: %d, error: %.4f" % (best_c.candidate, best_c.fitness[1], best_c.fitness[0]))
	
	return

def main() :
	
	# TODO nice logging using the 'log' module
	# TODO nice 'argparse' command-line arguments

	random_seed = 42
	data_file = "data/eureqa-example.csv"
	target_variable = "y"
	
	# parameters specific to the EA
	pop_size = 1000
	max_generations = 10
	
	print("Reading data from \"%s\"..." % data_file)
	data = read_csv(data_file)
	all_variables = list(data)
	all_variables.remove(target_variable)
	other_variables = all_variables
	target_variable_data = data[target_variable].as_matrix().ravel()
	other_variables_data = data[other_variables].as_matrix()
	print("Target variable \"" + target_variable + "\", data:", target_variable_data.shape)
	print("Other variables \"" + str(other_variables) + "\", data:", other_variables_data.shape)

	# parameters specific to the _Program trees
	function_names = ('add', 'sub', 'mul', 'div', 'sin', 'cos')
	function_set = [ functions._function_map[x] for x in function_names ]
	arities = [ f.arity for f in function_set ]
	parsimony_coefficient = 0 # this will probably be hard-coded, as parsimony is not necessary when going multi-objective
	init_depth = (2, 6)
	init_method = 'half and half'
	n_features = other_variables_data.shape[1]
	const_range = (-1., 1.)
	#metric_name = 'mean absolute error'
	metric_name = 'mse'
	metric = fitness._fitness_map[metric_name]
	p_point_replace = 0.01
	sample_weight = np.ones((other_variables_data.shape[0],))

	print("Initializing EA...")
	# unfortunately, inspyred requires a random.Random object; while gplearn needs a np.random.RandomState object
	# so, the (ugly) solution I found is to use TWO objects, initialized with the same random seed...
	prng = random.Random()
	prng.seed( random_seed )
	random_state = np.random.RandomState( random_seed )
	
	nsga2 = inspyred.ec.emo.NSGA2(prng)
	nsga2.observer = pse_observer 
	nsga2.variator = [pse_variator]
	nsga2.terminator = inspyred.ec.terminators.generation_termination
	nsga2.selector = inspyred.ec.selectors.tournament_selection
	
	final_pop = nsga2.evolve(
			generator=pse_generator,
			pop_size=pop_size,
			num_selected=pop_size,
			maximize=False,
			max_generations=max_generations,
			evaluator=pse_evaluator,
			
			# extra arguments, will be passed to the functions inside the "args" dictionary
			# arguments related to data
			data=data,
			target_variable=target_variable,
			other_variables=other_variables,
			target_variable_data=target_variable_data,
			other_variables_data=other_variables_data,
			
			# arguments specific to the _Program class
			function_set=function_set,
			arities=arities,
			init_depth=init_depth,
			init_method=init_method,
			n_features=n_features,
			const_range=const_range,
			metric=metric,
			p_point_replace=p_point_replace,
			parsimony_coefficient=parsimony_coefficient,
			sample_weight=sample_weight,
			random_state=random_state,
		)	
	
	
	return

if __name__ == "__main__" :
	sys.exit( main() )
