# ToDo List
Features to be implemented

Verificar se a solucao é otima, isto é se existem coeficientes negativos na equacao Z

Prepare the overall structure and create new branch for each functionality
- Change the user input data. Pass as parameters.








1. Prepare objective function (times -1)
2. Add slack variables (variáveis de folga) to constraints
3. Create the first table (matrix) with inputed elements
4. Choose pivot column and pivot row
5. Choose pivot number
6. Method to create tables at each iteration (a list of tables maybe?)
7. Minimizaton method (-1 * Z)
8. Transfer the create table method to the inside of Pyplex class

Second Release
1. Change input integer to float
2. Re-create table function with float items
3. Minimize method

Third Release
1. Big M method (when you have '=' on constraints)
2. Read values from text file (see np.loadtxt())

