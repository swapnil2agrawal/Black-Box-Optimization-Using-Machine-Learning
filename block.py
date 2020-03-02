"""
File Name : block.py
Author : Swapnil_Agrawal
Date Created : 07/01/2019
Python Version : 3.6
Detail : Implementing cyclic coordinate search algorithm for a given black box function and its approximation f(x) 
Input : C_file_name (3pk.c), input.in, output.out, number of data points, previously known optimum value

"""

import matplotlib.pyplot as plt
import sobol_seq
import numpy as np
import os
import sys
import re
import argparse
import sklearn   
from sklearn import linear_model
from scipy.optimize import minimize


if __name__ == '__main__':
    compileFile = sys.argv[1]
    inputFile = sys.argv[2]
    z = [] 
    x = []
    numofVar = 0
    outputFile = sys.argv[3]
    block_size = 5

	
def get_number(file):

    """ read .c file and match the line of for loop to get
    the number of variables
    output: (integer) number of variables """

    numOfVar = -1
    infile = open(file+'.c','r')
    for line in infile.readlines():
        matchObj = re.match(r'for',line,re.M|re.I)
        if matchObj:
            line_group1 = line.split(';')
            line_group2 = line_group1[1].split('<')
            numOfVar = line_group2[1].strip()
    return int(numOfVar)


def input_create(k,  file, numOfVar, x, UB, LB):

    """
    description - returns an input file with numofVar variables in it and 
    only kth row varying. (other fixed)
    A = matrix for keping rows other than kth rows fixed
    b = a number to keep the previously calculated optima (i.e. k-1th point)
    k = direction along which to vary (will be mulitple in this case)
    """

    infile = open(file, 'w')
    #print("data", A, b, k, numOfVar)

    for i in range(numOfVar):

        if(i in k):
            # generate a random number for kth row
            rn = 100*np.random.rand(1)[0]
            #print(rn)
            a = sobol_seq.i4_sobol(1, rn)[0][0]*(UB[i] - LB[i]) + LB[i]
            infile.write(str(a) + " ")
        

        else: 
            # enter constant for rest of the rows
            infile.write(str(x[i]) + " ")
            #print("why")

    infile.close()



def datafit(reg, X, Z): 

    l = len(Z)
    l1 = int(4*l/5)
    # l1 = int(2*l/3) # traning vs test dataset

    # Using 1/3rd of the data for fitting
    reg.fit(X[:l1], Z[:l1]) 

    # using rest of the dataset as a test set
    X1, z1 = X[l1:], Z[l1:]
    R = reg.score(X1, z1)
    # print(reg.coef_[0])
    # print(reg.intercept_)
    
    return R , reg.coef_, reg.intercept_


def generate_data(x):
    
    """ convert x (which contains all the input data in raw form) 
    to X (input data array) """
    
    X = []
    i = 0 

    for a in x:
    
        b = a.split()
        X.append([])

        for B in b:
            X[i].append(float(B))

        i = i+1

    return X


def output_generator(X):

    out = []
    with open(inputFile, 'w') as f:
        for x in X:
            f.write(str(x) + " ")

    os.system('.\\'+compileFile)

    outfile = open(outputFile,'r')

    for lines in outfile.readlines():
        out.append(float(lines.strip()))

    return out

def fitting(K, numOfVar, x1, UB, LB):

    """
    Given Kth row and b as previously found point of optima, it generates an input file with kth row varying 
    and rest all constant. Then it fit these points using linear regression"""
    x = []
    z = []
    reg = linear_model.LinearRegression()
    #print("Starthey")
    r = - 0.3
    while (r < 0.3 and len(x) < 100):
        for i in range(N):

            input_create(K, inputFile, numOfVar, x1, UB, LB)
            os.system('.\\'+compileFile)

            infile = open(inputFile,'r')

            for lines in infile.readlines():
                x.append(lines)

            outfile = open(outputFile,'r')

            for lines in outfile.readlines():
                z.append(float(lines.strip()))
            #print("StopHey")
            #print(x)
            # using only kth row in fitting a model

        X  = np.array(generate_data(x))

        X1 = []

        for i in K:
            X1.append(X[:, i])

        # print(np.shape(X1))

        # using quadratic terms for fitting
        X2 = []
        for i in range(len(X1)):
            X2.append(X1[i])
            X2.append(X1[i]**2)
        # X2 = np.vstack((X1, X1**2))
        # print(np.shape(X2))

        #print(X1)
        X2 = np.array(X2)
        # print(X2.T)
        # print(z)
        r, coeff, bias = datafit(reg, X2.T, np.array(z))

    return r, coeff, bias

 
def func(x, coeff, bias):

    y = bias
    i = 0 
    while( i < block_size):
        y = y + coeff[i]*x[i] + coeff[i+1]*x[i]**2
        i = i+2

    return y	



def minima(x0, coeff, bias, UB, LB):
    
    def c1(x):
        return -x+UB

    def c2(x):
        return x-LB

    constraint = [{'type': 'ineq', 'fun' : c1}, 
                {'type': 'ineq', 'fun' : c2}]

    ans = minimize(func, x0, args = (coeff, bias), constraints = constraint)
    return ans.x


def main(N, UB, LB, file, z_star):

    
    numOfVar = get_number(compileFile)
    init_r = UB - LB
    UB = np.ones(numOfVar)*UB
    LB = np.ones(numOfVar)*LB
    print("numOfVar:", numOfVar)
    print(UB)
    print(LB)
    #print(type(numofVar))
    os.system('gcc '+compileFile+'.c -o '+compileFile)
    
    feval = 0   
    # initial radnom value of optima 
    Z = 500
    tol = 0.05
    #print(abs(Z - z_star))
    # to generate guess for previous number at first iteration
    rn = 40*np.random.rand(1)[0]   
    #print(rn)
    #print(sobol_seq.i4_sobol(2, rn)[0])
    if(numOfVar > 40):
        div = int(numOfVar/40)
        num = list(sobol_seq.i4_sobol(40, rn)[0])
        for i in range(div):
            a = list(np.array(num)*(i + 2))
            # print("hello", a)
            for j in range(len(a)):
                num.append(a[j])
        # print("xn:", num, np.shape(num))
        xm = num[:numOfVar]*(UB - LB) + LB 
    else:
        xm = sobol_seq.i4_sobol(numOfVar, rn)[0]*(UB - LB) + LB 

    #x0 = sobol_seq.i4_sobol(1, rn)[0][0] 
    print(xm)
    track = []

    f = open(file, "w") 
    f.write("x[0]")
    f.write(",")
    f.write("x[1]")
    f.write(",")
    f.write("Z_blackbox")
    f.write(",")
    f.write("Z_Optimum")
    f.write(",")
    f.write("UB")
    f.write(",")
    f.write("UB")
    f.write(",")
    f.write("LB")
    f.write(",")
    f.write("LB")
    f.write("\n")
    
    # to count how many consecutive times minima is not changing (count_bad)
    countb = 0
    # to count how many consecutive times minima is improving (count_improvement)
    counti = 0 

    #while(feval < 50):
    continuousi = 0
    continuousb = 0

    while(abs(Z-z_star) > 0.0001):

        # stop if value of Z is not changing for 3 continuous iterations  
        # or if number of function evaluation is less than 50
        
        # after we have looped through all the variables, start from the beginning  
        ex = 0
        if(feval > numOfVar and ex < 1):
            #after it has optimized wrt to all variables, we would like to shrink search region
            dx = (UB - LB)[0]
            # to halve the range
            UB = xm + dx/4
            LB = xm - dx/4 
            # just to make sure to run this just once
            ex = ex+1

        if(counti > numOfVar):

            dx = (UB - LB)
            # to halve the range
            if(dx.all()>=0.1):
                UB = xm + dx/4
                LB = xm - dx/4 
            counti = 0 

        if(countb > numOfVar):

            dx = (UB - LB)

            # try to find better condition
            if(dx[0]<=init_r/8): 
                # make a jump
                # xm = xm + init_r/2
                UB = xm + init_r/4
                LB = xm - init_r/4
            else:
                # to double the range
                UB = xm + dx
                LB = xm - dx 
            countb = 0 

        countr = 0 
        feval = feval + 1
        print("feval:", feval)

        print("UB:", UB)
        print("LB:", LB)

        while(countr < numOfVar and countb <= numOfVar and counti <= numOfVar):
 
            # plt.scatter(xm[0], xm[1])
            # plt.pause(0.05)

            # stop if value of Z is within a tolerance of Z_star 
            # or if we have looped through all the variables
            
            K = np.random.randint(0, high = numOfVar, size = (block_size))
            print(K)
            #K = [1, 2]
            # to remove badly fitted models
            r = -0.3
            while(r < 0.3):
                print(r)
                r, coeff, bias = fitting(K, numOfVar, xm, UB, LB)
                        
            print("K:", K)
            print("coefficient of determination" , r)
            print("equation coefficient", coeff)
            print("bias", bias)
 
            # point of optima for a quadratic equation
            # to make sure this is also within bounds
            print(xm[K])
            xm[K] = minima(xm[K], coeff, bias, UB[K], LB[K])
            #xm[K] = -coeff[0]/(2*coeff[1]) 
            print("x0", K, xm)

            # value at x0 from model function
            Z_cap = func(xm[K], coeff, bias) 
            # actual value at x0 from blackbox function
            Z_cap2 = output_generator(xm)

            print("Z_cap:", Z_cap)
            print("Z_blacbox:", Z_cap2)

            countr = countr + 1
            
            # if improvement in minima value, update and also update improvement count
            if(Z_cap2[0] < Z):
                print("hey")
                Z = Z_cap2[0]
                if(continuousi == 1):
                    counti = counti + 1
                continuousi = 1
                continuousb = 0
                countb = 0


            # if no improvement, update countb (so that we can change bounds)
            else:
                print("heyo")
                if(continuousb == 1):
                    countb = countb + 1
                continuousb = 1
                continuousi = 0
                counti = 0 

            print("counti: ", counti)
            print("countb:", countb) 

            print("Z:", Z)

            t = np.zeros(10)
            t[0] = xm[0]
            t[1] = xm[1]
            t[2] = Z_cap2[0]
            t[3] = Z
            t[4] = UB[0]
            t[5] = UB[1]
            t[6] = LB[0]
            t[7] = LB[1]
            t[8] = counti
            t[9] = countb
            #t[8] = xm[2]
            track.append(t)

            for i in range(10):
                f.write(str(t[i]))
                f.write(",")

            f.write("\n")


    #plt.show()

    return Z

N = int(sys.argv[4])
UB = float(sys.argv[5])
LB = float(sys.argv[6])
data = sys.argv[7]
z_star = float(sys.argv[8])
main(N, UB, LB, data, z_star)
