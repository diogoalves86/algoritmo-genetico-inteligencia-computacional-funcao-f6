from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
from typing import List,Callable,Tuple
import math, sys
import random
from random import randrange
from decimal import Decimal

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome],int]
PopulateFunc = Callable[[],Population]
SelectionFunc = Callable[[Population,FitnessFunc], Tuple[Genome,Genome]]
CrossoverFunc = Callable[[Genome,Genome],Tuple[Genome,Genome]]
MutationFunc = [[Genome],Genome]


def function_f6 (x,y):
    x_y_pow = x ** 2 + y ** 2
    square_of_pow = math.sqrt(x_y_pow)
    print("square_of_pow: " + str(square_of_pow) + "\n")
    sin_pow = math.sin(square_of_pow) ** 2

    print("sin_pow: " + str(sin_pow) + "\n")

    fraction = Decimal(((Decimal(sin_pow) - Decimal(0.5))) / Decimal(((1 + (Decimal(0.001) * (x_y_pow)))) ** 2))

    print("fraction: " + str(fraction) + "\n")
    result = Decimal(0.5) - fraction
    print("result: " + str(format(result, ".60f")) + "\n")
    return result



#Gera o genoma aleatorio com tamanho k
def generate_genome(length: int) -> Genome:
    return choices([0,1],k=length)


#chama essa função multiplas vezes até a população ter o tamanho desejado
def generate_population(size:int,genome_length:int) -> Population:
    a = [generate_genome(genome_length) for _ in range(size)]
    return [generate_genome(genome_length) for _ in range(size)]


#funão fitness específica para o problema da função F6
def fitness(genome:Genome):
    #Transformação do genoma aleatorio em numeros X e Y para entrar na função F6
    cromossomox = ""
    cromossomoy = ""
    # for i in range(genome.length):
    for i in range(len(genome)):
        if i < len(genome)/2:
            cromossomox += str(genome[i])
        else:
            cromossomoy += str(genome[i])
    

    x = Decimal(int(cromossomox, 2))
    y = Decimal(int(cromossomoy, 2))

    # print("x: " + str(x) + "\ny: " + str(y) + "\n")
    
    ans = function_f6(x, y)
    # print("ans: "+ str(format(ans, ".4f")))

    #Ranqueamento do resultado
    for i in range(len(genome)):
        if ans == 1:
            print("deu bom \n")
            return 99999
        else:
            print("deu ruim \n")
            return abs(1/ans)

def selection_pair(population: Population, fitness_func: FitnessFunc)->Population:
    return choices(
        population=population,
        weigths =[fitness_func(genome) for genome in population],
        k=2
    )


def single_point_crossover(a:Genome, b:Genome) -> Tuple[Genome,Genome]:
    if len(a) != len(b):
        raise ValueError("Genomas a e b tem que ter mesmo tamanho")
    length = len(a)
    if length < 2:
        return a,b
    p = randint(1,length-1)
    return a[0:p] + b[p:],b[0:p] + a[p:]


def mutation (genome: Genome, num: int=1, probability: float = 0.5) -> Genome:
    for _ in range (num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index]-1)
    return genome


def run_evolution (
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int=100,
) -> Tuple[Population,int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range (int(len(population)/2)-1):
            parents = selection_func(population,fitness_func)
            offspring_a,offspring_b = crossover_func(parents[0],parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a,offspring_b]

        population = next_generation

    population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
    )
    return population,i

population,generations = run_evolution(
    populate_func = partial(
        generate_population,size=10, genome_length=44
    ),
    fitness_func = partial(
        fitness
    ),
    fitness_limit=740,
    generation_limit=100
)






