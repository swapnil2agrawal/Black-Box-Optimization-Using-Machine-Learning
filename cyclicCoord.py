"""
File Name : cyclicCoord.py
Author : Swapnil_Agrawal
Date Created : 06/25/2019
Python Version : 3.6
Detail : Implementing cyclic coordinate algorithm for a given black box function and its approximation f(x) 
Input : C_file_name (3pk.c), input.in, output.out, number of data points, output file for writing fitted model

"""

"""
Steps - 
1. pick all the dimension one-by-one and minimize along that one keeping other constant
2. repeat until termination condition is met

"""

import ML_algo
import os
import baron
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sobol_seq

if __name__ == '__main__':
	compileFile = sys.argv[1] # C source file to be complied
	inputFile = sys.argv[2]   # input arguments by random/sobol sampling 
	outputFile = sys.argv[3]      # output generated from the black box using the input points
	N = int(sys.argv[4])     # number of points to be used
	z_star = float(sys.argv[5]) #optimum value

	#answerFile = sys.argv[5]  # writing coefficient to a file

	numOfVar = ML_algo.get_number(compileFile)

	#r, coeff, bias = ML_algo.main(N, compileFile, inputFile, outputFile, 1)

	#print("really", coeff[0]/(2*coeff[0 + numOfVar]))

def func(x):
	X = np.hstack((x, x**2))
	y = bias + np.dot(X, coeff) 
	return y	


def outputGen(data):
	z = []

	# compile file
	os.system('gcc '+compileFile+'.c -o '+compileFile)

	# creating input file
	file = open(inputFile, 'w')
	for d in data:
		file.write(str(d) + " ")

	# run file
	os.system('.\\'+compileFile)

	# read output from file
	outfile = open(outputFile,'r')
	for lines in outfile.readlines():
		z.append(float(lines.strip()))

	return z

print(outputGen([2, 1]))

def input_gen(k, numOfVar, N):

	""" generate dataset for kth direction
	k = kth axis along which to vary
	numofvar = total number of variables
	N = total number of points to be generated"""
	X = np.zeros([N, numOfVar])
	z = np.zeros(N)

	rn = 100*np.random.rand(1)[0]
	print(rn)
	A = sobol_seq.i4_sobol(numOfVar, rn)[0]
	rn = 100*np.random.rand(1)[0]
	print(rn)
	B = sobol_seq.i4_sobol(N, rn)[0]

	for i in range(numOfVar):
		print(np.shape(X.T[i]))
		if(i==k):
			X.T[i] = B

		else:
			X.T[i] = A[i]

	# for i in range(N):
	# 	z[i] = outputGen(X[i])

	return X


def inputGen(k):

    # compile .c file
    #os.system('gcc '+compileFile+'.c -o '+compileFile)
    #print("compile-1")
    
    #x = []
    z = []

    X = input_gen(k, numOfVar, N)
    #print(X)

    

    for i in range(N):

    	print("array", i, ":", X[i])

    	file = open(inputFile, 'w')

    	for d in X[i]:
    		file.write(str(d) + " ")

    	os.system('gcc '+compileFile+'.c -o '+compileFile)
    	os.system('.\\'+compileFile)

    	outfile = open(outputFile,'r')

    	for lines in outfile.readlines():
    		z.append(float(lines.strip()))

    return X, z
# def main(x1, tol):

# 	# z = func(x1)
# 	# optimum = z
# 	feval = 0
# 	#print(x1)

# 	while(abs(z - z_star) > tol and feval < 10):

# 		k = 0
# 		print("feval", feval)
# 		while(k < numOfVar):


# 			# finding optimum in k-th direction
# 			p = coeff[k]/(2*coeff[k + numOfVar])
# 			x1[k] = p
# 			# print("hey", coeff[k]/(2*coeff[k + numOfVar]))
# 			# print(x1[k])
# 			# # updated function value
# 			z = func(x1)

# 			# stop if near optimal value found, else continue search in other directions
# 			if(abs(z - z_star) < tol):
# 				optimum = z
# 				break 
# 			else:
# 				k = k + 1

# 			print(x1, z)

# 		feval = feval + 1

# 	return optimum

def main():
	# z = []

	# os.system('gcc '+compileFile+'.c -o '+compileFile)
	# os.system('.\\'+compileFile)

	# outfile = open(outputFile,'r')

	# for lines in outfile.readlines():
	# 	z.append(float(lines.strip()))

	# print(z)

	# x = np.linspace(-300, 300)
	# y = np.zeros(len(x))

	# for i in range(len(x)):
	# 	y[i] = outputGen([x[i], 2])
	# plt.plot(x, y)
	# plt.show()
	#print(outputGen([0.7734375, 0.1484375]))
	print(inputGen(0))
	# x, z = input_gen(1, numOfVar, N)
	# print(x)
	# print(z)
	#return x, z
# x0 = np.array([2])

# print(main(x0, 0.2))

main()
