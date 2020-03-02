
## Black-Box Optimization Using Machine Learning

## Objective

The objective of this project is to perform black box optimization using machine learning and optimization methods.

<b>Constraint</b>
- Limited function evaluations

<b>Given</b>
- Limit on function calls
- The number of total variables
- Variable Bounds
- A black-box function to find input-output pairs

<b>Black Box</b> : Black box refers to a system which is so complex that it is easy to experiment rather than understanding it. In some setups, the function is not at all available in the algebraic form though we have access to function output on providing it some input. 

<b> Motivation </b>
- Wide Application 
- Evaluating the function can be expensive in terms of money or time
- Systems becoming more complex – difficult to understand than experiment
- In some cases, the function is not at all available in the algebraic form though we have access to function output on providing it some input (Example – Aspen process simulation)


![blackbox_Ms.png](/MS_Images/blackbox_Ms.png)

## Approach

For solving black-box optimization problem, my overall approach is, select a region, sample few points in that region using a sampling method which are described later in this [report](\MS_Report_Swapnil). These points can then be fitted to an algebraic function using machine learning models. The fitted model is optimized using baron. The process is repeated
until we have reached a stopping criteria or have achieved convergence.

Using different combination of sampling methods, sampling region and the update rules gives us different final results. In this project, I experimented with five different sampling methods and three different methods for region selection, namely global optimization method, trust-region method and cyclic coordinate search method. For developing surrogate model, Machine learning regression methods from Sci-kit learn packages are used. 

![appraoch_Ms.png](/MS_Images/appraoch_Ms.png)

## Problem Set

Non-Convex Smooth black-box problems are selected for analysis of the algorithm. These problems are further divided in four categories based on the number of input variables in the problem to measure the effect of dimensionality on the solution. 

A black-box function camel6 is used for determining the hyperparameters. Cmel6 is a two-dimensional test problem, which exhibits six local minima, two of which are global minima. The detailed discussion of this problem is mentioned in [Rios, Sahinidis paper](https://link.springer.com/article/10.1007/s10898-012-9951-y)

Te black-box problem set can be downloaded from [here](http://archimedes.cheme.cmu.edu/?q=dfocomp)

## Setup

<b>Requirements</b>
- Python 3.7
- Jupyter Notebooks
- UiPath
- Scikit-learn
        
<b>Instructions</b>
- Download the python files block.py, cyclicCoord.py and final.py. Download the data set from the link mentioned in the problem set and run using command line interface. 

## References

- Jonggeol Na, Youngsub Lim, and Chonghun Han. A modified direct algorithm for hidden constraints in an lng process optimization. Energy, 126:488–500, 2017
    
- Luis Miguel Rios and Nikolaos V. Sahinidis. Derivative-free optimization: a review of algorithms and comparison of software implementations. Journal of Global Optimization, 56(3):1247–1293, Jul 2013.

- Zhiwei Qin, Katya Scheinberg, and Donald Goldfarb. Efficient block-coordinate descent algorithms for the group lasso. Mathematical Programming Computation, 5(2):143–169, 2013.
    
- Stephen J Wright. Coordinate descent algorithms. Mathematical Programming, 151(1):3–34, 2015.

