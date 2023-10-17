# EXERCISE #5 PERCEPTRON
# AUTHOR: Samantha Shane C. Dollesin
# STUDENT NO.: 2020-01893
# SECTION: WX-1L
# PROGRAM DESCRIPTION: This program takes an input file, performs linear classification using perceptrons and writes the results on an output file.

import numpy as np
import pandas as pd

#this function writes an iteration onto the output file
def add_to_file(file, iteration, matrix):
    file.write("Iteration: "+ str(iteration)+"\n")
    file.write(matrix.to_string() + "\n")

input = open("input.txt", "r")

if (input.readable()):
    lr = float(input.readline().strip())  #learning rate
    th = float(input.readline().strip())  #threshold
    b = float(input.readline().strip())   #bias

    #x and z values
    values = input.read().split("\n")
    for i in range(len(values)):
        values[i] = (values[i].split(" "))
        for j in range(len(values[i])):
            values[i][j] = int(values[i][j])

    input.close()
    output = open("output.txt", "w")

    no_of_rows = len(values)            #number of rows
    no_of_x = len(values[i][:-1])       #number of x variables

    #create column names based on the number of x variables
    col_names = []
    for x in range(no_of_x): col_names.append("x"+str(x))
    col_names.append("b")
    for w in range(no_of_x): col_names.append("w"+str(w))
    col_names.extend(["wb", "a", "y", "z"])

    #initialize the table
    tb  = []

    for i in range(no_of_rows):
        tb.append([])
        for y in range(no_of_x):
            tb[i].append(values[i][y])      #append x values
        tb[i].append(b)                     #append bias
        for x in range(len(values[i])+2):
            tb[i].append(0)                 #append initial weights and values of a and y (all are 0)
        tb[i].append(values[i][-1])         #append z
    
    tb.append([0] * len(tb[0])) #add last row for final adjusted weights

    y = len(tb[0])-2    #index of y column
    a = len(tb[0])-3    #index of a column
    converged = False
    iteration = 1
    
    while (not converged):
        print("Iteration: ", iteration)
        for row in range(no_of_rows+1):
            #adjust weights if not the first row
            if (row > 0):
                for w in range(no_of_x+1):
                    tb[row][w+no_of_x+1] = tb[row-1][w+no_of_x+1] + (lr*tb[row-1][w]*(tb[row-1][-1]-tb[row-1][y]))     #compute for adjusted weight
                    # if(row == 1): print(tb[row-1][w+no_of_x+1], " + ", lr , " * ", tb[row-1][w], " * ", tb[row-1][-1], " - ", tb[row-1][y], " = ", tb[row][w+no_of_x+1])

            for x in range(no_of_x+1):
                tb[row][a] = tb[row][a] + (tb[row][x]*tb[row][x+no_of_x+1])     #compute for a
                # if(row == 1): print(tb[row][x], " * ", tb[row][x+no_of_x+1], " = ",tb[row][x]*tb[row][x+no_of_x+1] )
            
            tb[row][y] = 1 if (tb[row][a] >= th) else 0    #determine value of y
        
        matrix = pd.DataFrame(np.matrix(tb), columns=col_names)
        print(matrix)
        add_to_file(output, iteration, matrix)
        iteration = iteration + 1

        equal = []
        for x in range(no_of_x+1):
            weights = []
            for row in range(1,no_of_rows+1):                
                weights.append(tb[row][x+no_of_x+1])    #collect all xn weights in a list     
            equal.append(all(w == weights[0] for w in weights))     #check if xn's weights are all equal

        if all(equal):
            converged = True    #if weights of all x variables have converged, stop iterating
        else:
            for x in range(no_of_x+1):
                tb[0][x+no_of_x+1] = tb[no_of_rows][x+no_of_x+1]      #replace initial values
            for row in range(no_of_rows):
                tb[row][a] = 0     #reset y and a values to 0
                tb[row][y] = 0