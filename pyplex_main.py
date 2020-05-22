import platform  # For getting the operating system name
import subprocess  # For executing a shell command
import numpy as np
import sys, ast, getopt

# import ast, getopt, sys, copy, os

"""
Equation options
"""
GREATER_THEN = 1
EQUAL = 2
LESS_THEN = 1
EQUATION_OPTIONS = (
	(GREATER_THEN, '>='),
	(EQUAL, '='),
	(LESS_THEN, '<='),
)


class PyplexTableau():

	def __init__(self, number_decisions, number_constraints, variables, constraints):
		self.table = np.full((number_constraints + 1, number_decisions + number_decisions + 1),0)
		# Inserting data
		self.table[0] = variables
		for i in range(len(constraints)):
			self.table[i + 1] = constraints[i]
		self.table_columns = list()
		self.table_rows = list()

	def __str__(self):
		return self.table

	def print_tableau(self):
		self.num_rows = self.table.shape[0]
		self.num_columns = self.table.shape[1]
		# Columns names
		print('\t'.join(self.table_columns))
		for r in range(self.sum_rows):
			print(self.table_rows[i] + ' ')
			column = ''
			for c in range(self.num_columns):
				column += self.table[i][j] + '\t'
			print(column)


class PyplexSolver():

	def __init__(self, first_table, max_min='max', verb=False):
		# Holds the table for all the iterations (for debug and verbose purpose)
		self.simplex_iter = list()
		self.simplex_iter.append(first_table)
		self.pivot_number = 0
		self.max_min = max_min
		# if true will print every iteration
		self.verbose = verb


	# Returns the index of the minimum value of the first row/line
	def next_pivot_column(self, table):
		next_pvt = np.where(table[0] == np.min(table[0]))
		return next_pvt[0][0] if len(next_pvt[0]) >= 1 else -1

	def next_pivot_row(self, table, pivot_col):
		column_pivot = np.array(table[0:, pivot_col], dtype=float)
		# Returns the last column, the result column
		column_r = np.array(table[0:, -1:], dtype=float)
		column_r_trans = column_r.transpose()
		with np.errstate(divide='ignore', invalid='ignore'):
			pivot_line = np.true_divide(column_r_trans, column_pivot)
			pivot_line[~ np.isfinite(pivot_line)] = 0  # -inf inf NaN
			pivot_line[pivot_line == -0] = 0

		value = np.min(pivot_line[np.nonzero(pivot_line)])
		next_pvt = np.where(pivot_line[0] == value)
		return next_pvt[0][0] if len(next_pvt[0]) >= 1 else -1

		# #np.min(c[np.nonzero(c)])
		#
		#
		# #np.divide(cr,cp, out=np.zeros(cr.shape, dtype=float), where=cp!=0)
		# # np.where(line_pivot[0] == np.min(line_pivot[0]))
		# # import pdb
		# # pdb.set_trace()
		# # return next_pvt[0][0] if len(next_pvt) > 1 else -1
		# return next_pvt[0][0] if len(next_pvt[0]) >= 1 else -1

	def exec_minimize(self):
		print('Minimize')
		# Z*-1
		# exec_maximize

	def exec_maximize(self):
		print('Maximize')
		print(self.simplex_iter[0])
		pivot_c = self.next_pivot_column(self.simplex_iter[0])
		print('Pivot Column: {}'.format(pivot_c))
		pivot_r = self.next_pivot_row(self.simplex_iter[0], pivot_c)
		print('Pivot Row: {}'.format(pivot_r))


	def print_results(self):
		clear_screen()
		print("Results:")

	def create_table(self):
		pass
		# return numpy table

	def exec_solver(self, ):
		if self.max_min.lower() == 'min':
			self.exec_minimize()
		else:
			self.exec_maximize()


# Creates an matrix/table with zeros
def create_matrix(num_col, num_rows):
	table = np.full((num_rows, num_col),0)
	return table

def clear_screen():
	"""
	Clears the terminal screen.
	"""
	# Clear command as function of OS
	command = "cls" if platform.system().lower() == "windows" else "clear"
	# Action
	return subprocess.call(command) == 0


def welcome_message():
	clear_screen()
	print('=' * 30)
	print('\tP Y P L E X')
	# print('\')
	print('=' * 30)


def print_equation_options():
	eq_options = 'Choose on of the following: \n'
	for i in EQUATION_OPTIONS:
		eq_options += "{} for '{}', ".format(i[0], i[1])
	eq_options = eq_options[:-2]
	print(eq_options)


# if __name__ == "__main__":
# 	print_equation_options()
def read_decision_vars():
	decisionVars = dict()

	# Read from the user decision variables
	numDecisionVar = int(input("Number of decision variables in the problem: "))
	for i in range(numDecisionVar):
		value = int(input("Value of Decision Variable X{}: ".format(i + 1)))
		decisionVars.update({'X{}'.format(i + 1): value})
	return decisionVars


def print_decision_vars(vars):
	obj_function = ''
	for i in vars:
		obj_function += '{}{}'.format(vars[i], i) + ' + '

	# removes last plus sigh and extra space character
	obj_function = obj_function[:-2]
	print('\tObjective Function: '+obj_function)


def read_constraintis(num_decision_var):
	print('Constraint Variables:')
	numConstraints = int(input("\nNumber of constraints: "))
	constraint_list = list()

	for i in range(numConstraints):
		constraint = list()
		print('Constraints #{}: '.format(i+1))
		for i in range(num_decision_var):
			value = int(input("\tValue of X{}: ".format(i + 1)))
			constraint.append(value)
		print_equation_options()
		eq_option=int(input())
		constraint.append(eq_option)
		value = int(input("Value of equation: "))
		constraint.append(value)

		# Adds a single constraint to a list of constraints
		constraint_list.append(constraint)
	return constraint_list


def print_constraints(constraints):
	for con in constraints:
		single_constraint = ''
		for j in range(len(con)):
			single_constraint += '{}X{} + '.format(con[j], j+1)
		single_constraint = single_constraint[:-2]
		print(single_constraint)


def print_help_parameters():
	print('Options are')
	print('pyplex.py -d <vector-decision_variables> -A <matrix-constraints> -r <vector> -t <obj_func_type>')
	print('\th: Prints this help')
	print('\td: Objective function coefficients')
	print('\tA: Matrix of the constraints (coefficients)')
	print('\tr: Result of the constraints equation (Ax <= r )')
	print('\tt: Type of objective function (max or min)')


def create_empty_matrix(rows, cols):
	table = np.full((rows, cols),0)
	return table


def generate_first_table(dec_vars, const, result):
	# First row is the obj. function
	# deci_vars = np.array(dec_vars)
	# deci_vars *= -1

	first_row = np.append(dec_vars,np.full((1,len(const)+1),0))
	first_row *= -1
	# Creates a matrix with the constraints coef.
	const_var = np.array(const)
	# Generates the slacks matriz
	slacks_var = np.eye(len(const))
	# Join both the
	table = np.column_stack((const_var, slacks_var))
	# Attach the result at the far end column
	table1 = np.column_stack((table, result))
	return np.vstack((first_row, table1))


if __name__ == "__main__":

	welcome_message()
	decision_vars = []
	constraints_matrix = []
	result_equation = []
	type_obj_function = ''
	verbose = False

	# First argument is the application's name (pyplex.py)
	argv = sys.argv[1:]
	try:
		options, args = getopt.getopt(argv, "hd:A:r:t:v", ["d=", "A=", "r=", "t=", "v"])
	except getopt.GetoptError:
		print(
				'pyplex.py -d <vector-decision_variables> -A <constraints-matrix> -r <vector> -t <obj_func_type> ' +
				'-v <verbose-True-False>'
		)
		sys.exit(2)
	for opt, arg in options:
		if opt == '-h':
			print_help_parameters()
			sys.exit()
		elif opt in ("-d"):
			decision_vars = ast.literal_eval(arg)
		elif opt in ("-A"):
			constraints_matrix = ast.literal_eval(arg)
		elif opt in ("-r"):
			result_equation = ast.literal_eval(arg)
		elif opt in ("-t"):
			type_obj_function = arg.strip()
		elif opt in ("-v"):
			verbose = arg.strip()


	if not decision_vars or not constraints_matrix or not result_equation:
		print('Insufficient or invalid parameters. Please provide correct arguments.')
		print_help_parameters()
		sys.exit()

	# If not provided, the we assume that is maximization
	if type_obj_function not in ('max', 'min'):
		type_obj_function = 'max'

	init_table = generate_first_table(decision_vars,constraints_matrix,result_equation)

	my_solver = PyplexSolver(init_table,type_obj_function,verbose)
	my_solver.exec_solver()


# b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
# Matrix com variaveis de folga
# g = np.eye(3)
# np.column_stack((d, g))


	#
	# decision_variables = list()
	# constraint_list = list()
	#
	# decision_variables = read_decision_vars()
	# print_decision_vars(decision_variables)
	#
	# constraint_list= read_constraintis(len(decision_variables))
	# print_constraints(constraint_list)

# number_col=len(tabela[0,:])
# number_row=len(tabela[:,0])

# m = gen_matrix(2,2)
# constrain(m,'2,-1,G,10')
# constrain(m,'1,1,L,20')
# obj(m,'5,10,0')
# print(maxz(m))
#
# m = gen_matrix(2,4)
# constrain(m,'2,5,G,30')
# constrain(m,'-3,5,G,5')
# constrain(m,'8,3,L,85')
# constrain(m,'-9,7,L,42')
# obj(m,'2,7,0')
# print(minz(m))
