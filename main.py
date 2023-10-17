# EXERCISE #5 PERCEPTRON
# AUTHOR: Samantha Shane C. Dollesin
# STUDENT NO.: 2020-01893
# SECTION: WX-1L
# PROGRAM DESCRIPTION: 

import numpy as np

def print_2d(tb):
    for row in tb:
        print(row)
    print()

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

    no_of_rows = len(values)

    print("learning rate: ", lr)
    print("threshold: ", th)
    print("bias: ", b)
    print_2d(values)

    #initialize the table
    tb  = []
    no_of_x = len(values[i][:-1])                 #number of x variables

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
                    #if(row == 2): print(tb[row-1][w+no_of_x+1], " + ", lr , " * ", tb[row-1][w], " * ", tb[row-1][-1], " - ", tb[row-1][y], " = ", tb[row][w+no_of_x+1])

            for x in range(no_of_x+1):
                tb[row][a] = tb[row][a] + (tb[row][x]*tb[row][x+no_of_x+1])     #compute for a
                # if(row == 2): print(tb[row][x], " * ", tb[row][x+no_of_x+1], " = ",tb[row][x]*tb[row][x+no_of_x+1] )
            
            tb[row][y] = 1 if (tb[row][a] >= th) else 0    #determine value of y
        
        print("  x0  x1   b  w0  w1  wb  a   y  z")
        print(np.matrix(tb))
        iteration = iteration + 1

        equal = []
        for x in range(no_of_x+1):
            weights = []
            for row in range(1,no_of_rows+1):
                weights.append(tb[row][x+no_of_x+1])            
            equal.append(all(w == weights[0] for w in weights))

        if all(equal):
            converged = True
        else:
            for x in range(no_of_x+1):
                tb[0][x+no_of_x+1] = tb[no_of_rows][x+no_of_x+1]      #replace initial values
            for row in range(no_of_rows):
                tb[row][a] = 0     #reset y and a values to 0
                tb[row][y] = 0


    

        
        

