from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt
import math
from random import choices
import sys
from decimal import Decimal

# Parametros de entrada do GA
bounds = [[-100, 100], [-100, 100]]
iteration = 40
half_genome = 22
bits = 44  #numero de bits do Genoma
pop_size = 100
crossover_rate = 0.65
mutation_rate = 0.08
genome = []

def function_f6(I):
    x = I[0]
    y = I[1]
    x_y_pow = float(x ** 2 + y ** 2)
    square_of_pow = math.sqrt(x_y_pow)
    sin_pow = (math.sin(square_of_pow) ** 2)
    fraction = (((sin_pow) - (0.5))) / (((1 + (float(0.001) * (x_y_pow)))) ** 2)
    result = (0.5) - fraction
    return result

# -----------------------------------------------------------------------------------------------#

def crossover(pop, crossover_rate):
    offspring = list()
    for i in range(int(len(pop)/ 2)):
        p1 = pop[2 * i - 1].copy()  # parent 1
        p2 = pop[2 * i].copy()  # parent 2
        if rand() < crossover_rate:
            cp = randint(1,len(p1)-1,size=2)
            while cp[0] == cp[1]:
                cp = randint(1,len(p1)-1,size=2)
            cp = sorted(cp)
            c1 = p1[:cp[0]] + p2[cp[0]:cp[1]] + p1[cp[1]:]
            c2 = p2[:cp[0]] + p1[cp[0]:cp[1]] + p2[cp[1]:]
            offspring.append(c1)
            offspring.append(c2)
        else:
            offspring.append(p1)
            offspring.append(p2)
    return offspring


def mutation(pop, mutation_rate):
    offspring = list()
    for i in range(int(len(pop))):
        p1 = pop[i].copy() # parent
        if rand() < mutation_rate:
            cp = randint(0,len(p1)) #gera gene aleatorio
            c1 = p1
            if c1[cp] == 1:
                c1[cp] = 0 #flip
            else:
                c1[cp] = 1
            offspring.append(c1)
        else:
            offspring.append(p1)
    return offspring


# Seleção do metodo da roleta
def selection(pop, fitness, pop_size):
    next_generation = list()
    elite = np.argmax(fitness)
    next_generation.append(pop[elite])     #mantem o melhor
    P = [f/sum(fitness) for f in fitness]  #seleção da prob
    index = list(range(int(len(pop))))
    index_selected = np.random.choice(index,size=pop_size-1,replace=False,p=P)
    s=0
    for j in range(pop_size-1):
        next_generation.append(pop[index_selected[s]])
        s+=1
    return next_generation

#Cria a população de 44bits para real
def inicializa_pop(bounds,bits,genome):
    lower_x_boundary, upper_x_boundary = bounds[0]
    lower_y_boundary, upper_y_boundary = bounds[1]
    half_genome = bits/2
    population = []
    population_bit = []

    for i in range(pop_size):
        cromossomox = ""
        cromossomoy = ""

        for j in range(len(genome)):
            if j < half_genome:
                cromossomox += str(genome[j])
                # print("x: " + cromossomox)
            else:
                cromossomoy += str(genome[j])
                # print("y: " + cromossomoy)

        individual_bit = {
            "x": cromossomox,
            "y": cromossomoy
        }
        #print(cromossomox)

        cromossomox_int = int(bytearray(cromossomox, "utf8"), 2)
        cromossomoy_int = int(bytearray(cromossomoy, "utf8"), 2)
        #print(cromossomox_int)

        cromossomox_decimal = Decimal(cromossomox_int) * Decimal(
            (upper_x_boundary - lower_x_boundary) / (pow(2, half_genome) - 1)) + Decimal(lower_x_boundary)
        cromossomoy_decimal = Decimal(cromossomoy_int) * Decimal(
            (upper_y_boundary - lower_y_boundary) / (pow(2, half_genome) - 1)) + Decimal(lower_y_boundary)

        #print(cromossomox_decimal)

        individual = {
            "x": cromossomox_decimal,
            "y": cromossomoy_decimal
        }
        population.append(individual)
        population_bit.append(individual_bit)

        genome = generate_genome(bits)

    print(population_bit)
    sys.exit()
    return population,population_bit


def generate_genome(length: int):
    choices_selected = choices([0, 1], k=length)
    return choices_selected

# ---------------------------------Programa Principal-----------------------------------------------#

genome = generate_genome(bits)
pop = inicializa_pop(bounds,bits,genome)
best_fitness = []

for gen in range(iteration):
    offspring = crossover(pop,crossover_rate)
    offspring = mutation(pop, mutation_rate)

    for s in offspring:
        pop.append(s)

    real_chromossome = [inicializa_pop(bounds,bits,p) for p in pop]

    for d in real_chromossome:
        fitness = function_f6(d)   #fitness value

    index = np.argmax(fitness)
    current_best = pop[index]
    best_fitness.append(1/max(fitness)-1)
    pop = selection(pop,fitness,pop_size)


# ---------------------------------Gerando os Gráficos-----------------------------------------------#

#Falta Acertar o grafico para fazer a media dos experimentos

fig = plt.figure()
plt.plot(best_fitness)
fig.subtitle('Evolução')
plt.xlabel('Iteração')
plt.ylabel('Function F6 Values')
print("Solução Otima",inicializa_pop(bounds,bits,current_best))