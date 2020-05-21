import platform  # For getting the operating system name
import subprocess  # For executing a shell command
import numpy as np

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


class PyplexSolver():

	def __init__(self, ):
		# Holds the table for all the iterations (for debug and verbose purpose)
		self.simplex_iter = list()
		self.pivot = 0
		self.max_min = 'max'

	class PyplexTableau():

		def __init__(self, number_decisions, number_constraints, decisions, constraints):
			self.table = np.zeros(number_constraints+1, number_decisions+number_decisions+1)
			# Inserting data
			self.table[0] = decisions
			for i in range(len(constraints)):
				self.table[i+1]=constraints[i]
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




	def exec_minimize(self):
		print('Minimize')
		# Z*-1
		# exec_maximize

	def exec_maximize(self):
		print('Maximize')

	def print_results(self):
		clear_screen()
		print("Results:")

	def create_table(self):

		pass
		# return numpy table

	def exec_solver(self, decision_vars, constraint_vars, max_min='max'):


		if max_min.lower() == 'min':
			self.exec_minimize()
		else:
			self.exec_maximize()




# Creates an matrix/table with zeros
def create_matrix(num_col, num_rows):
	table = np.zeros((num_rows, num_col))
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
	decisionVars = list()

	# Read from the user decision variables
	numDecisionVar = int(input("Number of decision variables in the problem: "))
	for i in range(numDecisionVar):
		value = int(input("Value of Decision Variable X{}: ".format(i + 1)))
		decisionVars.update({'X{}'.format(i + 1): value})
	return decisionVars

if __name__ == "__main__":

	welcome_message()

	decisionVars = list()
	constraintVars = list()

	decisionVars = read_decision_vars()

	obj_function = ''
	for i in decisionVars:
		obj_function += '{}{}'.format(decisionVars[i], i) + ' + '

	# removes last plus sigh and extra space character
	obj_function = obj_function[:-2]
	print('\tObjective Function: '+obj_function)

	numConstraints = int(input("\nNumber of constraints: "))

	clear_screen()
	for i in range(numConstraints):
		constraint = list()
		print('Constraints #{}: '.format(i+1))
		for i in range(numDecisionVar):
			value = int(input("\tValue of X{}: ".format(i + 1)))
			constraint.append(value)
		print_equation_options()
		eq_option=int(input())
		constraint.append(eq_option)
		value = int(input("Value of equation: "))
		constraint.append(value)

		# Adds a single constraint to a list of constraints
		constraintVars.append(constraint)

	print('Constraint Variables:')
	print(decisionVars)

	clear_screen()




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
