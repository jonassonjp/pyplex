import platform  # For getting the operating system name
import subprocess  # For executing a shell command

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

def maximize():
	pass


# if __name__ == "__main__":
# 	print_equation_options()

if __name__ == "__main__":

	welcome_message()

	decisionVars = dict()
	constraintVars = list()

	# Determine Variables and Constraints
	numDecisionVar = int(input("Number of decision variables in the problem: "))
	for i in range(numDecisionVar):
		value = int(input("Value of Decision Variable X{}: ".format(i+1)))
		decisionVars.update({'X{}'.format(i+1) : value})

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
