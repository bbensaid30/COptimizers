# Coptimizers

## Goal of the project

I write a general fully-connected neural network in C++ as well as a lot of Deep Learning optimizers. 
A version using the Shaman Library is also available to control the numerical errors.

## Structure

* activations: implement a lot of classical activation functions with their derivatives
* classic: deterministic GD, Momentum, Adam-like optimizers with a stopping criteria on the gradient
* data: read data files and store them in a vector map
* incremental: new rebalanced splitting schemes (RAG and RAGL)
* init: random initializations 
* LMs: many variants of the Levenberg-Marquardt algorithm
* perso: Armijo backtracking optimizers
* perte: loss functions
* propagation: direct propagation, backpropagation and commputation of the quasi-hessian
* scaling: useful for some LM algorithms
* Sclassic: stochastic optimizers (SGD, RRGD, RRAdam, ...) with a deterministic stopping criteria
* Sperso: stochastic classical Armijo suggested by Vaswani
* Stest: test of stochastic algorithms on analytical benchmarks
* Stirage: run in parallel and store a lot of training results for stochastic optimizers (different initializations and seeds)
* Straining: the function to run a stochastic training
* test: test of deterministic algorithms on analytical benchmarks
* tirage: run in parallel and store a lot of training results for deterministic optimizers (different initializations)
* utilities: some useful functions 

## How to run it ?
A makefile is given as a example as well as for the shaman version. 
Some installations are needed: Eigen3, EigenRand and Shaman (well explained in the corresponding gits).
The folder "Mains" provide a lot of examples: how to use the test files, use real datasets, ... 

