import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
from random import randrange
from decimal import Decimal
import math, sys
import ga

def sphere(x):
    return sum(x**2)

#Definição do Problema
problem = structure()
problem.costfunc = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10


#Definição dos Parametros do GA
params = structure()
#numero de iterações
params.maxit = 100
#tamanho da população
params.npop = 20
params.pc = 1
params.gamma = 0.1
#taxa de mutação
params.mu = 0.1
params.sigma = 0.1

# Rodando Algoritmo Genetico
out = ga.run(problem,params)

#Plot results








