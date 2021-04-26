from ypstruct import structure
import numpy as np


#Algoritmo Genetico
def run(problem,params):

    # Problem information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    #Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    #Multiplicando por 2 para ter um numero par
    nc = int(np.round(pc*npop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma


    #Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    #BestSolution Ever found
    best_sol = empty_individual.deepcopy()
    best_sol.cost = np.inf

    #Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(0,npop):
        pop[i].position = np.random.uniform(varmin,varmax,nvar)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < best_sol.cost:
            best_sol = pop[i].deepcopy()

    # Best cost of iterations
    best_cost = np.empty(maxit)


    #Main Loop
    for it in range(0,maxit):
        #population of children
        popc = []

        #nc = number of children
        # nc//2 - Maximum number of crossover loop
        for k in range(nc//2):

            #Select Parents
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            #Performing Crossover
            c1,c2 = crossover(p1,p2,gamma)

            #Performing Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)


            #Aplicando as fronteiras (-10,10)
            apply_bound(c1,varmin,varmax)
            apply_bound(c2, varmin, varmax)


            #Evaluate First Offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < best_sol.cost:
                best_sol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < best_sol.cost:
                best_sol = c2.deepcopy()

            #Add offsprings to popc
            popc.append(c1)
            popc.append(c2)


        #Merge, Sort and Select
        pop = pop + popc
        #para cada x na população retorna x.cost - ordena população por x.cost
        pop = sorted(pop, key=lambda x: x.cost)

        #Seleciona os n membros da população
        pop = pop[0:npop]

        #Store cost of iteration
        best_cost[it] = best_sol.cost

        #Show Iteration Information
        #print("Iteration {}: Best Cost = {}".format(it,best_cost[it]))

    #Output
    out = structure()
    out.pop = pop
    out.best_sol = best_sol
    out.best_cost = best_cost
    return out

def crossover (p1,p2,gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma,1+gamma,*c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha * p2.position + (1 - alpha) * p1.position
    return c1,c2

#mu = mutation rate
#sigma = step size of real mutation
def mutate(x,mu,sigma):
    y = x.deepcopy()
    #arrays de True or Falses indicando as posicões das Mutações
    flag = (np.random.rand(*x.position.shape) <= mu)
    #Retorna os indices das posicoes que serao modificadas
    ind = np.argwhere(flag)
    y.position[ind] = y.position[ind] + sigma*np.random.randn(*ind.shape)
    return y


def apply_bound(x,varmin, varmax):
    x.position= np.maximum(x.position,varmin)
    x.position = np.minimum(x.position, varmax)


