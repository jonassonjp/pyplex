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
		self.constraint_limits = dict()

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
		"""
			two_phase_method: Resolving no standard PL problems in two phases
		"""

		print('Initiating two phase method')
		print('Phase 1')
		# Set initial tableau


		print('Phase 2')
		#change
		exit(-1)

	def exec_minimize(self):
		"""
			exec_minimize: Resolvs minimizing problems
		"""
		#Todo Minimize Implement Two Phase Method
		#Todo Minimize Z row must be the last row

		print("Minimize on it's way..." )

		# Checks for inequalities
		if 'L' in self.inequalities:
			print('Begin two phase method')
			exit(0)



	def exec_maximize(self,initial_tab):
		tableau_list = list()
		tableau_list.append(initial_tab)

		print('Initializing maximize method')
		i =0
		while self.optimality_check(tableau_list[i].table[0]):

			# Discover the pivot column
			pivot_c = self.next_pivot_column(tableau_list[i].table)
			if self.verbose: print('Pivot Column: {}'.format(pivot_c))

			# Discover the pivot row
			pivot_r = self.next_pivot_row(tableau_list[i].table, pivot_c)
			if self.verbose: print('Pivot Row: {}'.format(pivot_r))

			# Discover the pivot number
			pivot_number = tableau_list[i].table[pivot_r][pivot_c]
			if self.verbose: print('Pivot Number: {}'.format(pivot_number))

			# Create the new tableau
			new_tableau = tableau_list[i].copy()

			# Calculates the next round tableau
			new_tableau = self.next_round_tab(new_tableau, pivot_c, pivot_r, pivot_number)
			if self.verbose:
				print("Table: ")
				self.simplex_iter[i].print_tableau()

			# Adds the new tableau to the iteration list
			tableau_list.append(new_tableau)
			i += 1
		self.set_shadow_price(new_tableau)
		self.set_constraint_limits(new_tableau)
		return tableau_list # list of all iteraction


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
		col_width = 54
		print('=' * col_width)
		print('\t\tR E S U L T S')
		print('=' * col_width)

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
		self.print_sensitivity_analysis_report()


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

	def set_constraint_limits(self, last_tableau):
		b = last_tableau.table[1:,-1:]
		b = np.dot(-1,b)
		const_labels = last_tableau.table_columns_names
		const_lables_n_position = dict()
		const_lables_n_position = {const_labels[i]: i for i, s in enumerate(const_labels) if 'S' in s}
		y_as_ind = {i for i, s in enumerate(last_tableau.table_columns_names) if 'S' in s}
		S_as = last_tableau.table[1:, min(y_as_ind):-1]
		delta_b = np.full((len(y_as_ind),1),0)



		for i, key in enumerate(const_lables_n_position):
			limit_values = list()
			delta_b[i][0] = 1
			S_Deltab = np.dot(S_as,delta_b)
			result_b = np.zeros((len(y_as_ind),1))
			for k, Bi in enumerate(S_Deltab):
				result_b[k] = b[k][0] / Bi
			limit_values.append(np.max(result_b))
			limit_values.append(np.min(result_b))
			delta_b = np.full((len(y_as_ind), 1), 0)
			self.constraint_limits[key] = limit_values



	def print_sensitivity_analysis_report(self):

		# rhs_range = dict()
		# tableau = self.simplex_iter[-1]
		# for i in range(len(self.decision_var)):
		# 	rhs_range[tableau.table_columns_names[i]] = [1,2]
		# print(rhs_range)
		# exit(0)
		final_tableau = self.simplex_iter[-1]
		results = final_tableau.table[1:,-1:]
		if self.max_min == 'max':
			dec_var = 'X'
			dec_vars_list = final_tableau.table_rows_names

		elif self.max_min == 'min':
			dec_var = 'Y'
			dec_vars_list = final_tableau.table_columns_names

		all_vars_list = final_tableau.table_columns_names
		all_const_list = final_tableau.table_columns_names

		# Create a dictionary with all the decision variables set to 0
		all_dec_final_value = dict()
		all_dec_final_value = {all_vars_list[i]: 0 for i, s in enumerate(all_vars_list) if dec_var in s}
		all_dec_final_value = {i: all_dec_final_value[i] for i in sorted(all_dec_final_value)}

		# Create a dictionary with all the constraints variables set to 0
		all_const_final_value = dict()
		all_const_final_value = {all_const_list[i]: 0 for i, s in enumerate(all_const_list) if 'S' in s}
		all_const_final_value = {i: all_const_final_value[i] for i in sorted(all_const_final_value)}

		# Slack, Suplus Variables if any
		slack_variables = {dec_vars_list[i]: i for i, s in enumerate(dec_vars_list) if 'S' in s}
		slack_variables = {i: slack_variables[i] for i in sorted(slack_variables)}

		# Grab all the decision variables from the last tableau
		decision_in_solution = {dec_vars_list[i]: i for i, s in enumerate(dec_vars_list) if dec_var in s}

		if self.max_min == 'max':
			for key, value in decision_in_solution.items():
				all_dec_final_value[key]=final_tableau.table[value][-1]

			for key, value in slack_variables.items():
				all_const_final_value[key]=final_tableau.table[value][-1]
		elif self.max_min == 'min':
			for key, value in decision_in_solution.items():
				all_dec_final_value[key]=final_tableau.table[0][value]
			for key, value in slack_variables.items():
				all_const_final_value[key]=final_tableau.table[0][value]

		decision_result=list()
		for i, key in enumerate(all_dec_final_value):
		# for key, value in all_dec_final_value.items():
			line=list()
			line.append(key+ '\t    |')
			line.append(str(all_dec_final_value[key])+'\t    |')
			line.append('0' + '\t    |') # custo reduzido
			line.append(str(self.decision_var[i])+ '\t    |')
			# Acrescimo
			line.append(str(0.0)+'\t     |')
			# Decrescimo
			line.append(str(0.0)+'\t        |')
			decision_result.append(line)

		constraint_resul = list()
		for i, key in enumerate(all_const_final_value):
			line=list()
			line.append(key+ '\t    |')      # Constraint label
			line.append(str(all_const_final_value[key])+'\t    |') # Constraint value
			line.append(str(self.shadow_price[key]) + '\t    |') # shadow price | preço sombra
			line.append(str(results[i][0])+ '\t    |') # Result ( 'b' )
			# Acrescimo
			line.append(str(self.constraint_limits[key][0])+'\t     |') # ToDo Implement ==> Not Done yet
			# Decrescimo
			line.append(str(self.constraint_limits[key][1])+'\t        |') # ToDo Implement ==> Not Done yet
			constraint_resul.append(line)



		clear_screen()
		width_column = 97
		print('\n')
		print('=' * width_column, '\n\t\tR E L A T Ó R I O\tS E N S I B I L I D A D E')
		print('-' * width_column)
		print('Var Decisao |\tValor Final |\tCusto Reduz | Coef.Objetivo | Acrs. Possível | Decres. Possível |')
		for val in decision_result:
			print('\t'.join(map(str, val)))
		# For valores aqui
		print('-' * width_column)
		print('\nRestrições')
		print('-' * width_column)
		print(' Restrição  |\tValor Final |\tPreço Sombra | Result. (LD) | Acrs. Possível | Decres. Possível |')
		for val in constraint_resul:
			print('\t'.join(map(str, val)))
		print('-' * width_column)



	def convert_gaussian(self, tableau):
		converted_tableau = tableau

		row_elem_name =  tableau.table_rows_names
		col_elem_name =  tableau.table_columns_names
		# Get the index of all variabels in the row
		vars_row = {row_elem_name[i]: i for i, s in enumerate(row_elem_name) if s != 'Z'}
		# Get the index of all variables in the columns
		vars_col  = {col_elem_name[i]: i for i, s in enumerate(col_elem_name) if s != 'b'}

		# Build a dictionary, where the key is the row label, and the value is the element (aij)
		# in which needs to turn into 1
		elements = dict()
		selected_cols=list()
		for key in vars_row:
			elements[vars_row[key]]=(vars_row[key], vars_col[key])
			selected_cols.append(vars_col[key])

		# Scanning the table for the elements we need to turn into 1
		for key, value in elements.items():
			pivot_element = tableau.table[value]
			if pivot_element != 1:
				new_row = tableau.table[key]
				pivot_element = np.full((1,len(new_row)),tableau.table[value],dtype=float)
				new_row = np.divide(new_row, pivot_element)
				converted_tableau.table[key] = new_row

		# Going through all possible columns
		# elements (dict) key=row_index, values=aij
		# example: 1: (2,3)
		for key, value in elements.items():
			selec_col_ind = value[1]
			col_elements = converted_tableau.table[:,selec_col_ind]  # value[1] is the column from aij element
			pivot_line = converted_tableau.table[value[0],:]  # value[1] is the column from aij element
			for index, col_value in np.ndenumerate(col_elements):
				# check is its the pivot value
				if value == (index[0],selec_col_ind):
					continue
				# Only if the column value is not 0 or 1
				if col_value not in (0,1):
					old_line = converted_tableau.table[index]
					multi_element = np.full(len(old_line), col_value, dtype=float)
					new_line = old_line - np.multiply(multi_element,pivot_line)
					converted_tableau.table[index] = new_line

		return converted_tableau

	def sensibility_analysis(self, final_tableau_original, decision_vars, constraints_coef, result):

		# Slack var values (final tableau) y*
		y_as_ind = {i for i, s in enumerate(final_tableau_original.table_columns_names) if 'S' in s}
		y_as = np.array([final_tableau_original.table[0][i] for i in y_as_ind])

		# Z* Z_as = y_as * b_mod
		b_mod = np.transpose(result)
		Z_as = np.dot(y_as, b_mod)

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

		width_column = 60
		print('\n\n')
		print('=' * width_column, '\n\t\tS E N S I B I L I D A D E')
		print('-' * width_column)

		print('  Tableu Final Original: ')
		print('.' * width_column)
		final_tableau_original.print_tableau()
		print('\n')
		print('-' * width_column)
		print('  Tableu Inicial Modificado: ')
		print('.' * width_column)
		tableau_revised.print_tableau()

		# Converts the tableau to the Gaussian form
		converted_tableau = self.convert_gaussian(tableau_revised)
		print('\n')
		print('-' * width_column)
		print('  Convertida para a forma apropriada: ')
		# print('  Converted to Apropriate form: ')
		print('.' * width_column)
		converted_tableau.print_tableau()

		# Checks for non negative values in 'b' and in 'Z'
		if self.optimality_check(converted_tableau.table[0:, -1:]):
			elem = converted_tableau.table[0:, -1:]
			neg_elem = elem[elem<0]
			print('\n *** Solução nao viável pois possui elemento(s) negativo(s): {}'.format(neg_elem))

		# and self.optimality_check(converted_tableau.table[0]):

		# coef_z_line = converted_tableau[0]
		# # Remove the last element, leaving just the coefficients
		# coef_z_line = np.delete(coef_z_line, len(coef_z_line) - 1)
		# # Check if all elements in this line is 0
		# indexes = [i for i, x in enumerate(coef_z_line) if x != 0]

	def read_user_values(self):
		values = dict()
		cont_read = True
		while cont_read:
			print('Enter the  values separating them by colons.')
			print('Example: 1,2,3')
			dec_vars = list(map(int,input("Decision variables: ").strip().split(',')))
			numb_const = int(input('How many constrains: '))
			const = list()
			for i in range(numb_const):
				const.append(list(map(int,input("Constraint #{}: ".format(i+1)).strip().split(','))))
			results = list(map(int,input("Results {}(right hand side): ".format(numb_const)).strip().split(',')))
			values['dec_vars'] = dec_vars
			values['const'] = const
			values['result'] = results

			print('Values:')
			for key, value in values.items():
				print('\t{} = \t{}'.format(key, values[key]))
			confirm_ok = str(input('Confirm values?([Y]|n): ') or "Y")
			cont_read = False if confirm_ok in ('Y','y') else True
		return values

	def exec_solver(self, ):
		"""
			exec_solver: Does all the magic
			Minimization problems are already transformed in the PyPlex constructor
		"""
		# if self.max_min.lower() == 'min':
		# 	self.simplex_iter = self.exec_minimize(self.simplex_iter[0])
		# else:
		# 	self.simplex_iter = self.exec_maximize(self.simplex_iter[0])

		self.simplex_iter = self.exec_maximize(self.simplex_iter[0])
		self.print_results()

		resp = None
		while resp not in ('s', 'n', 'p'):
			resp = str(input('\nDeseja fazer alguma alteração nos valores? (S/N): ') or 'p').lower()

		# Preset values, testing case
		decision_var = [4, 5]
		constraints = [[1, 0], [0, 2], [2, 2]]
		result = [4, 24, 18]

		if resp == 's':
			input_values = self.read_user_values()
			decision_var = input_values['dec_vars']
			constraints = input_values['const']
			result = input_values['result']
		elif resp == 'n':
			exit(0)

		self.sensibility_analysis(
			self.simplex_iter[-1],      # last tableau
			decision_var,               # new decision variables
			constraints,                # new constrains
			result                      # new results
		)

		# ToDo  Implement read values from user input
		#   Read values
		#   Call sensibility_analysis


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
	col_width = 54
	print('=' * col_width)
	print('\t\tP Y P L E X')
	# print('\')
	print('=' * col_width)


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
	print('\ti: Inequalities (E, L, G)')
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
				'pyplex.py -c <vector-decision_variables> -A <constraints_coef> -i <inequalities> -b <vector> -p <obj_func_type> ' +
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
