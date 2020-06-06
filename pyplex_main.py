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

	def __init__(self,  dec_vars, const, ineq, result, max_min='max', verb=False):
		# Holds the table for all the iterations (for debug and verbose purpose)
		self.simplex_iter = list()
		self.decision_var = dec_vars
		self.constraints = const
		self.inequalities = ineq
		self.result = result
		self.max_min = max_min
		# if true will print every iteration
		self.verbose = verb
		self.shadow_price = list()

		self.sensi_analysis_iter = list()

		if max_min == 'min':
			#Transpose var and set to decision var and constraints
			self.decision_var = result
			self.result = dec_vars
			self.constraints = np.transpose(np.array(self.constraints))
			self.constraints = self.constraints.tolist()

		first_tableau = self.generate_first_tableau(self.decision_var, self.constraints, self.inequalities, self.result)
		self.simplex_iter.append(first_tableau)
		self.pivot_number = 0




	def generate_first_tableau(self, dec_vars, const, ineq, result):
		"""
			Generate the first table with all the values
			First row is the obj. function
		"""
		self.decision_var=dec_vars
		self.constraints=const
		self.inequalities=ineq
		tableau = PyplexTableau(len(dec_vars),len(const))
		tableau.table_rows_names.append('Z')
		for x in range(1, len(dec_vars)+1):
			tableau.table_columns_names.append('X{}'.format(x))

		if self.max_min == 'max':
			column_label = 'S{}'
		elif self.max_min == 'min':
			column_label = 'Y{}'

		for x in range(1, len(const)+1):
			tableau.table_columns_names.append(column_label.format(x))
			tableau.table_rows_names.append(column_label.format(x))

		tableau.table_columns_names.append('b')

		# Appends 0 to the rest of the line
		first_row = np.append(dec_vars, np.full((1, len(const) + 1), 0))
		# First row  Z * -1
		first_row *= -1

		# Creates a matrix with constraints coef.
		const_var = np.array(const)

		# Generates the slacks/surplus matrix
		slacks_var = np.eye(len(const))

		# Check for inequalities - adding surplus variables insted of slack
		# indexes = self.check_inequalities(self.inequalities, 'G')
		# if self.max_min == 'max' and (len(indexes)>0):
		# 	# *-1
		#
		# indexes = self.check_inequalities(self.inequalities, 'L')
		# if self.max_min == 'min' and (len(indexes)>0):
		# 	# *-1


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

	def check_inequalities(self, ineq, value):
		indexes = [i for i, x in enumerate(ineq) if x == value]
		return indexes

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
		column_b = np.array(table[0:, -1:], dtype=float)
		column_b_trans = column_b.transpose()

		pivot_line = self.div_array(column_b_trans, pivot_column)

		# Search this line minus the first element, Z value
		pivot_line_minus_Z = np.delete(pivot_line, 0)

		try:
			# Value is the minimum POSITIVE value excluding the first value which is Z line
			value = np.min(pivot_line_minus_Z[pivot_line_minus_Z > 0])
		except ValueError:
			print('\nThere is no solution. \nSomething is wrong. Maybe you need to reformulate your problema')
			sys.exit(2)

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

	# Convert the inequalities to stand form
	def convert_to_standard(self, inequalities):
		pass

	def optimality_check(self, elements):
		"""
			Checks if the Z's row elements has negative values. Is so, returns True, otherwise False
		"""
		return True if len(elements[elements < 0]) > 0 else False


	def two_phase_method(self, dec_vars, constraints, right_hand_side):
		print('Initiating two phase method')
		print('Phase 1')
		# Set initial tableau


		print('Phase 2')
		#change
		#self.exec_maximize()

	def exec_minimize(self):
		print("Minimize on it's way..." )

		# Checks for inequalities
		if 'L' in self.inequalities:
			print('Begin two phase method')
			exit(0)

		# # We will create a new first table for solving Minimizing problems
		# new_tableau = self.simplex_iter[0].copy()
		# # Swap the Z line with the last one
		# temp_Z_line=np.array(new_tableau.table[0], dtype=float)
		# new_tableau.table[0] = new_tableau.table[-1]
		# new_tableau.table[-1] = temp_Z_line
		#
		#
		# # Swap the Z line label the last one
		# temp_row_label = new_tableau.table_rows_names[0]
		# new_tableau.table_rows_names[0] = new_tableau.table_rows_names[-1]
		# new_tableau.table_rows_names[-1] = temp_row_label
		#
		# # Transpose the new created matrix
		# new_tableau.table = np.transpose(new_tableau.table)
		#
		# new_tableau.num_rows = np.size(new_tableau.table, 0)
		# new_tableau.num_columns = np. size(new_tableau.table, 1)
		#
		# # Creates the labels for the new tableau
		# new_tableau.table_rows_names = list()
		# new_tableau.table_columns_names_names = list()
		# for x in range(1, len(self.decision_var)+1):
		# 	new_tableau.table_columns_names.append('X{}'.format(x))
		#
		# for x in range(1, len(self.constraints)+1):
		# 	new_tableau.table_columns_names.append('S{}'.format(x))
		# 	new_tableau.table_rows_names.append('S{}'.format(x))
		# new_tableau.table_rows_names.append('Z')
		#
		# new_tableau.table_columns_names.append('b')
		#
		#
		# self.simplex_iter[0] = new_tableau

		#ToDo Minimize
		# DONE: Swap the Z line with the last one
		# DONE: Swap the Z line label the last one
		# Create the new labels
		# Transpose the matrix
		# Add the matrix to the list of iterations
		# Maximize the matrix
		# See results
		# Maybe needs to invert the lines again




		#Todo Minimize Implement Two Phase Method
		#Todo Minimize Z row must be the last row


	def exec_maximize(self):
		# Number of columns
		number_col=self.simplex_iter[0].num_columns
		# number_row=len(tabela[:,0])

		print('Maximize')
		i =0

		while self.optimality_check(self.simplex_iter[i].table[0]):

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
		self.set_shadow_price(self.simplex_iter[-1])

	def print_optimal_solution(self):
		# Gets the last table in the iteration list
		final_tableau = self.simplex_iter[-1]

		if self.max_min == 'max':
			dec_var = 'X'
			dec_vars_list = final_tableau.table_rows_names
		elif self.max_min == 'min':
			dec_var = 'Y'
			dec_vars_list = final_tableau.table_columns_names

		# Grab all the desicion variables from the last tableau
		decision_in_solution = {dec_vars_list[i]: i for i, s in enumerate(dec_vars_list) if dec_var in s}

		# Slack, Suplus Variables if any
		other_variables = {dec_vars_list[i]: i for i, s in enumerate(dec_vars_list) if 'S' in s}

		# Order the decision variables
		decision_in_solution = {i: decision_in_solution[i] for i in sorted(decision_in_solution)}

		# order the Slack, Suplus Variables
		other_variables = {i: other_variables[i] for i in sorted(other_variables)}

		print('Z\t= {:.2f}'.format(final_tableau.table[0][-1]))

		if self.max_min == 'max':
			for key, value in decision_in_solution.items():
				print('{}\t= {:.2f}'.format(key, final_tableau.table[value][-1]))
			# Prints the other varibles in the solution
			for key, value in other_variables.items():
				print('{}\t= {:.2f}'.format(key, final_tableau.table[value][-1]))
		elif self.max_min == 'min':
			for key, value in decision_in_solution.items():
				print('{}\t= {:.2f}'.format(key, final_tableau.table[0][value]))
			# Prints the other varibles in the solution
			for key, value in other_variables.items():
				print('{}\t= {:.2f}'.format(key, final_tableau.table[0][value]))




	def print_results(self):
		clear_screen()
		print('=' * 30)
		print('\tR E S U L T S')
		print('=' * 30)

		print("Original matrix:")
		print("Iteration #0")
		self.simplex_iter[0].print_tableau()
		for i in range(1, len(self.simplex_iter)):
			print("\nIteration #{}".format(i))
			self.simplex_iter[i].print_tableau()
		# First entry of simplex_iter is the initial tableau, so it does'nt count.
		print('Total Iterations: {}'.format(len(self.simplex_iter)))
		if self.max_min == 'max':
			print('Maximization problem')
		elif self.max_min == 'min':
			print('Minimization problem')
		print('\nOptimal Solution: ')
		self.print_optimal_solution()

		# shadow_price = self.extract_shadow_price()
		# print('\nShadow price: \t'.join(shadow_price))

		# ToDo Vê uma forma se vai perguntar se deseja imprimir os resultados agora
		self.print_sensitivity_analysis()


	def set_shadow_price(self, last_tableau):
		shadow_var = 'S' if self.max_min == 'max' else 'Y'
		shadow_var_list = last_tableau.table_columns_names

		# Grab all the desicion variables from the last tableau
		shadow_var_ind = {shadow_var_list[i]: i for i, s in enumerate(shadow_var_list) if shadow_var in s}

		# Order the dictionary
		shadow_var_ind = {i: shadow_var_ind[i] for i in sorted(shadow_var_ind)}

		# Sets the value according to the last tableau
		shadow_var_value = dict()
		for key, value in shadow_var_ind.items():
			shadow_var_value[key] = last_tableau.table[0][value]
		self.shadow_price = shadow_var_value




	def print_sensitivity_analysis(self):

		# rhs_range = dict()
		# tableau = self.simplex_iter[-1]
		# for i in range(len(self.decision_var)):
		# 	rhs_range[tableau.table_columns_names[i]] = [1,2]
		# print(rhs_range)
		# exit(0)

		clear_screen()
		width_column = 90
		print('=' * width_column, '\n\t\tR E L A T Ó R I O\tS E N S I B I L I D A D E')
		print('-' * width_column)
		print('Var Decisao | \tFinal Value |\tCusto Reduz | Coef.Objetivo | Acrs. Possível | Decres. Possível |')
		# For valores aqui
		print('-' * width_column)
		print('-' * width_column, '\n\nRestrições')
		print('-' * width_column)
		print(' Constraint  | \tFinal Value |\tShadow Price  | Rest. Lado Dir | Acrs. Possível | Decres. Possível |')
		for i in range(len(self.constraints)):
			print('\tC{}\t'.format(i+1), end='')
			# ToDo falta extrair slack
			print('{:.2f}-slack\t'.format(self.result[i]), end=' ')
			# print('{:.2f}'.format(self.shadow_price[i]), end=' ')
			print('{:.2f}'.format(self.result[i]), end='')
			# ToDo falta calcular
			print('{:.2f}'.format(00000), end='')
			# ToDo falta calcular
			print('{:.2f}'.format(00000))

		# print('Var Decisao\tValor\tPreço Somb\tRest. Lado Dir\tAcrs. Possível\tDecres. Possível')



	def sensibility_analysis(self, final_tableau_original, decision_vars, constraints_coef, result):



		# Slack var values (final tableau) y*
		y_as_ind = {i for i, s in enumerate(final_tableau_original.table_columns_names) if 'S' in s}
		y_as = np.array([final_tableau_original.table[0][i] for i in y_as_ind])

		# Z* Z_as = y_as * b_mod
		b_mod = np.transpose(result)
		Z_as = np.dot(y_as,b_mod)



		rows_const_array = len(constraints_coef)
		cols_const_array = len(constraints_coef[0])
		A_mod = np.array(constraints_coef)
		c_mod = final_tableau_original.table[0, 0:cols_const_array]

		# Calc objective function coef. (y_as * A_mod - c_mod)
		c_calc = np.dot(y_as, A_mod) - decision_vars

		S_as = final_tableau_original.table[1:, min(y_as_ind):-1]

		# Matrix from constraint vars
		A_as = np.dot(S_as, A_mod)

		# Result b* = S* x b_mod
		b_as = np.dot(S_as, b_mod)
		b_as = b_as.reshape(len(b_as),1)

		# Build the matrix
		tableau_revised = final_tableau_original.copy()
		tableau_revised.table[0][-1] = Z_as                         # Z*
		tableau_revised.table[0][0:len(decision_vars)] = c_calc     # c
		tableau_revised.table[1:, 0:len(decision_vars)] = A_as      # A*
		tableau_revised.table[1:, -1:] = b_as                       # b*

		print('-' * 30)
		print('Tableu Final Original: ')
		final_tableau_original.print_tableau()
		print('\n')
		print('-' * 30)
		print('Tableu Inicial Modificado: ')
		tableau_revised.print_tableau()



		pass
		# sdadasd
		# Read New Values
		# Calc new Values
		# Build New Tableau
		# Calc simplex new tableau
		# Viability test
		# Optimaity test
		# Re-optimization
		# self.sensi_analysis_iter = list()



	def read_new_data(self):
		print("Ler dados")


	def exec_solver(self, ):
		# if self.max_min.lower() == 'min':
		# 	self.exec_minimize()
		# else:
		# 	self.exec_maximize()
		self.exec_maximize()
		# self.print_results()

		# ToDo  Ask if wants to change
		#   Read values
		#   Call sensibility_analysis
		decision_var = [ 4,5]
		constraints = [[1,0], [0,2],[2,2]]
		result = [4,24,18]
		self.sensibility_analysis(
			self.simplex_iter[-1],  # last tableau
			decision_var,           # new decision variables
			constraints,       # new constrains
			result             # new results
		)

		# resp = None
		# while resp not in ('s', 'n'):
		# 	resp = input('Deseja fazer alguma alteração nos valores? (S/N): ').lower()
		#
		# if (resp=='s'):
		# 	self.read_new_data()
		# 	# Processar dados, passando o último tableau
		# else:
		# 	exit(0)






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
	print('Options are:')
	print('pyplex.py -d <vector-decision_variables> -A <matrix-constraints> -r <vector> -t <obj_func_type>')
	print('\th: Prints this help')
	print('\tc: Objective function coefficients')
	print('\tA: Matrix of the constraints (coefficients)')
	print('\ti: Inequations (E, L, G)')
	print('\tb: Result of the constraints equation (Ax <= r )')
	print('\tp: Type of objective function (max or min)')
	print('\tv: Verbose mode. Prints out every iteration')


def create_empty_matrix(rows, cols):
	table = np.full((rows, cols),0)
	return table


if __name__ == "__main__":
	welcome_message()
	decision_vars = []
	constraints_coef = []
	result_equation = []
	inequalities = []
	type_obj_function = ''
	debug = ''
	verbose = True
	verb_arg = ''

	# First argument is the application's name (pyplex.py)
	argv = sys.argv[1:]
	try:
		options, args = getopt.getopt(argv, "hc:A:i:b:p:v:d", ["c=", "A=", "i=", "b=", "p=", "v=", "d="])
	except getopt.GetoptError:
		print(
				'pyplex.py -c <vector-decision_variables> -A <constraints_coef> -i <inequations> -b <vector> -p <obj_func_type> ' +
				'-v <verbose-True-False>'
		)
		sys.exit(2)
	for opt, arg in options:
		if opt == '-h':
			print_help_parameters()
			sys.exit()
		elif opt in ("-c"):
			decision_vars = ast.literal_eval(arg)
		elif opt in ("-A"):
			constraints_coef = ast.literal_eval(arg)
		elif opt in ("-b"):
			result_equation = ast.literal_eval(arg)
		elif opt in ("-i"):
			inequalities = arg.strip()
		elif opt in ("-p"):
			type_obj_function = arg.strip()
		elif opt in ("-v"):
			verb_arg = arg.strip()
		elif opt in ("-d"):
			debug = arg.strip()

	# if debug.lower() == 'true':
	# 	decision_vars = []
	# 	constraints_coef = []
	# 	result_equation = []
	# 	print("DEBUG mode - values are fixed")
	# 	sys.exit()

	verbose = True if verb_arg.lower() == 'true' else False

	if not decision_vars or not constraints_coef or not result_equation or not inequalities:
		print('Insufficient or invalid parameters. Please provide correct arguments.')
		print_help_parameters()
		sys.exit()

	# If not provided, the we assume that is maximization
	if type_obj_function not in ('max', 'min'):
		type_obj_function = 'max'

	simplex_solver = PyplexSolver(decision_vars, constraints_coef, inequalities, result_equation, type_obj_function, verbose)
	simplex_solver.exec_solver()


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