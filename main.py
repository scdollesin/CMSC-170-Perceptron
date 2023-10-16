# EXERCISE #5 PERCEPTRON
# AUTHOR: Samantha Shane C. Dollesin
# STUDENT NO.: 2020-01893
# SECTION: WX-1L
# PROGRAM DESCRIPTION: 

import numpy as np

input = open("input.txt", "r")
x= 5

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

    print("learning rate: ", lr)
    print("threshold: ", th)
    print("bias: ", b)
    print(values)

