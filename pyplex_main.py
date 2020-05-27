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

	def __init__(self, number_decisions=0, number_constraints=0):
		self.table = np.full((number_constraints + 1, number_decisions + number_constraints + 1),0)
		# self.num_rows = self.table.shape[0]
		self.num_rows = np.size(self.table, 0)
		self.num_columns = np. size(self.table, 1)
		self.table_columns_names = list()
		self.table_rows_names = list()

	def __str__(self):
		return self.table

	def copy(self):
		new_tableau = PyplexTableau()
		new_tableau.table = self.table.copy()
		new_tableau.num_columns = self.num_columns
		new_tableau.num_rows = self.num_rows
		new_tableau.table_columns_names = self.table_columns_names
		new_tableau.table_rows_names = self.table_rows_names
		return new_tableau


	def print_tableau(self):

		# Columns names
		print('\t' + '\t'.join(self.table_columns_names))
		column = ''
		for r in range(self.num_rows):
			# print(self.table_rows_names[r], end='')
			column = self.table_rows_names[r] + '\t'
			for c in range(self.num_columns):
				column = column + '{:.2f}'.format(self.table[r][c]) + '\t'
			print(column)


class PyplexSolver():

	def __init__(self,  dec_vars, const, result, max_min='max', verb=False):
		# Holds the table for all the iterations (for debug and verbose purpose)
		self.simplex_iter = list()
		first_tableau = self.generate_first_tableau(dec_vars,const,result)
		self.simplex_iter.append(first_tableau)
		self.pivot_number = 0
		self.max_min = max_min
		# if true will print every iteration
		self.verbose = verb
		self.decision_var=list()
		self.constraints=list()

	def generate_first_tableau(self, dec_vars, const, result):
		"""
			Generate the first table with all the values
			First row is the obj. function
		"""
		# self.decision_var=dec_vars
		# self.constraints=const
		tableau = PyplexTableau(len(dec_vars),len(const))
		tableau.table_rows_names.append('Z')
		for x in range(1, len(dec_vars)+1):
			tableau.table_columns_names.append('X{}'.format(x))

		for x in range(1, len(const)+1):
			tableau.table_columns_names.append('F{}'.format(x))
			tableau.table_rows_names.append('F{}'.format(x))

		tableau.table_columns_names.append('R')

		# Appends 0 to the rest of the line
		first_row = np.append(dec_vars, np.full((1, len(const) + 1), 0))
		# Row Z * -1
		first_row *= -1
		# Creates a matrix with the constraints coef.
		const_var = np.array(const)
		# Generates the slacks matrix
		slacks_var = np.eye(len(const))
		# Join both the constraints and slacks variables
		tableau.table = np.column_stack((const_var, slacks_var))
		# Attach the result at the far end column
		table1 = np.column_stack((tableau.table, result))
		tableau.table = np.vstack((first_row, table1))
		return tableau

	def div_array(self, array1, array2):
		with np.errstate(divide='ignore', invalid='ignore'):
			divided_array = np.true_divide(array1, array2)
			divided_array[~ np.isfinite(divided_array)] = 0  # remove -inf inf NaN
			divided_array[divided_array == -0] = 0   # remove -0
		return divided_array


	def next_pivot_column(self, table):
		"""
		Returns the index (coeficient) of the minimum value of the first row/line
		"""
		next_pvt = np.where(table[0] == np.min(table[0]))
		# rever essa condicao aqui
		# valor precisa ser > 0
		return next_pvt[0][0] if len(next_pvt[0]) >= 1 else -1

	def next_pivot_row(self, table, pivot_col_coef):
		"""
			Returns the index (coeficient) the next pivot row/line
			The next pivot row is the lowest number of the division of the result column (last)
			by the pivot column, BUT, we have to exclude the Z elements of both arrays.
		"""
		#ToDo refactor: pass only the array insted of whole table
		pivot_column = np.array(table[0:, pivot_col_coef], dtype=float)

		# Grab the result column (the last column in our table)
		column_r = np.array(table[0:, -1:], dtype=float)
		column_r_trans = column_r.transpose()

		pivot_line = self.div_array(column_r_trans, pivot_column)

		# Search this line minus the first element, Z value
		pivot_line_minus_Z = np.delete(pivot_line, 0)

		# Value is the minimum value excluding the first value which is Z line
		value = np.min(pivot_line_minus_Z[np.nonzero(pivot_line_minus_Z)])

		next_pvt = np.where(pivot_line[0] == value)

		# In case there is a draw (more then one minimum value) return the minor coefficient
		next_pvt = np.min(next_pvt)

		return next_pvt


	def next_round_tab(self, tableau, pivot_col_coef, pivot_row_coef, pivot_number):
		next_tableau = tableau
		table = tableau.table
		num_rows = next_tableau.num_rows
		num_cols = next_tableau.num_columns
		# num_rows = len(table[:,0])
		# num_cols = len(table[0, :])
		pivot_line = np.array(table[pivot_row_coef], dtype=float)
		pivot_numb_array = np.full((1, num_cols),pivot_number, dtype=float)
		pivot_line = np.divide(pivot_line, pivot_numb_array)
		for i in range(num_rows):
			# Creates an array with only pivot coef.
			pivot_coef = np.full((1, num_cols), table[i][pivot_col_coef], dtype=float)
			new_line = table[i]-(np.multiply(pivot_coef, pivot_line))
			table[i] = new_line

		# Restore original value for the pivot line
		table[pivot_row_coef] = pivot_line
		next_tableau.table = table
		next_tableau.table_rows_names[pivot_row_coef] = tableau.table_columns_names[pivot_col_coef]
		return next_tableau


	def check_negative_value_z(self, elements):
		"""
			Checks if the Z's row elements has negative values. Is so, returns True, otherwise False
		"""
		return True if len(elements[elements < 0]) > 0 else False

	def exec_minimize(self):
		print('Minimize Under constrtuction')


	def exec_maximize(self):
		# Number of columns
		number_col=self.simplex_iter[0].num_columns
		# number_row=len(tabela[:,0])

		print('Maximize')
		i =0
		while self.check_negative_value_z(self.simplex_iter[i].table[0]):

			if self.verbose: print(self.simplex_iter[i].print_tableau())
			# Discover the pivot column
			pivot_c = self.next_pivot_column(self.simplex_iter[i].table)
			if self.verbose: print('Pivot Column: {}'.format(pivot_c))

			# Discover the pivot row
			pivot_r = self.next_pivot_row(self.simplex_iter[i].table, pivot_c)
			if self.verbose: print('Pivot Row: {}'.format(pivot_r))

			# Discover the pivot number
			self.pivot_number = self.simplex_iter[i].table[pivot_r][pivot_c]
			if self.verbose: print('Pivot Number: {}'.format(self.pivot_number))

			# Create the new tableau
			new_tableau = self.simplex_iter[i].copy()
			if self.verbose:
				print("Next Iteration: ")
				self.simplex_iter[i].print_tableau()

			# Divide the new line by the pivot number
			new_pivot_line = self.div_array(
					new_tableau.table[pivot_r],
					np.full((1,number_col), self.pivot_number, dtype=float)
			)
			# table[pivot_r] = new_pivot_line
			if self.verbose: print("New pivot line: {}".format(new_pivot_line))

			new_tableau = self.next_round_tab(new_tableau, pivot_c, pivot_r, self.pivot_number)
			if self.verbose:
				print("Table: ")
				self.simplex_iter[i].print_tableau()

			self.simplex_iter.append(new_tableau)
			i += 1
			# Discover the pivot column
			pivot_c = self.next_pivot_column(self.simplex_iter[i].table)
			if self.verbose: print('Pivot Column: {}'.format(pivot_c))


	def print_optimal_solution(self):

		final_tableau = self.simplex_iter[-1]
		dec_vars_list = final_tableau.table_rows_names
		# Grab all the desicion variables from the last tableau
		decision_in_solution = {dec_vars_list[i]: i for i, s in enumerate(dec_vars_list) if 'X' in s}

		# Order the decision variables
		decision_in_solution = {i: decision_in_solution[i] for i in sorted(decision_in_solution)}
		print('Z\t= {:.2f}'.format(final_tableau.table[0][-1]))
		for key, value in decision_in_solution.items():
			print('{}\t= {:.2f}'.format(key, final_tableau.table[value][-1]))

	def print_results(self):
		clear_screen()
		print('=' * 30)
		print('\tR E S U L T S')
		print('=' * 30)

		print("Matriz original:")
		self.simplex_iter[0].print_tableau()
		for i in range(1, len(self.simplex_iter)):
			print("\nIteration #{}".format(i))
			self.simplex_iter[i].print_tableau()
		# First entry of simplex_iter is the initial tableau, so it does'nt count.
		print('Total Iterations: {}'.format(len(self.simplex_iter)))
		print('\nOptimal Solution: ')
		self.print_optimal_solution()



	def print_sensitivity_analysis(self):
		clear_screen()
		width_column=90
		print('=' * width_column, '\n\t\tR E L A T Ó R I O\tS E N S I B I L I D A D E')
		print('-' * width_column)
		print('Var Decisao | \tValor |\tCusto Reduz | Coef.Objetivo | Acrs. Possível | Decres. Possível |')
		# For valores aqui
		print('-' * width_column)
		print('-' * width_column, '\n\nRestrições')
		print('-' * width_column)
		print(' Restrição  | \tValor |\tPreço Somb  | Rest. Lado Dir | Acrs. Possível | Decres. Possível |')
		# print('Var Decisao\tValor\tPreço Somb\tRest. Lado Dir\tAcrs. Possível\tDecres. Possível')


	def create_table(self):
		pass
		# return numpy table

	def exec_solver(self, ):
		self.print_sensitivity_analysis()

		# if self.max_min.lower() == 'min':
		# 	self.exec_minimize()
		# else:
		# 	self.exec_maximize()
		# self.print_results()
		# value=input("Imprime Relatório de Sensibilidade (S/N)?: ")[0]
		# if value.lower() == 's':
		# 	self.print_sensitivity_analysis()



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
		eq_option = int(input())
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


if __name__ == "__main__":

	welcome_message()
	decision_vars = []
	constraints_matrix = []
	result_equation = []
	type_obj_function = ''
	verbose = True

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

	my_solver = PyplexSolver(decision_vars,constraints_matrix,result_equation,type_obj_function,verbose)
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
#
# for i in range(1, alunos + 1):
#     nota = input("Coloque a nota do aluno " + str(i) + ":" )
#     notas.append(nota)

#Como dividir por zero - testar
# c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)