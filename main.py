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

    tb  = []

    #for i in range((len(values[1])*2)+3):
    #    tb.append([])

    #initialize the table
    for i in range(no_of_rows):
        tb.append([])
        for y in range(len(values[i][:-1])):
            tb[i].append(values[i][y])      #append x values
        tb[i].append(b)                     #append bias
        for x in range(len(values[i])+2):
            tb[i].append(0)                 #append initial weights and values of a and y (all are 0)
        tb[i].append(values[i][-1])         #append z
    
    tb.append([0] * len(tb[0])) #add last row for final adjusted weights

    print(np.matrix(tb))
