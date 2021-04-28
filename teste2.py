import random
import math
import statistics
import sys
from random import choices
from decimal import Decimal
from ast import literal_eval

def generate_genome(length: int):
    choices_selected = choices([0, 1], k=length)
    return choices_selected

genome_list = []
def generate_population(pop_size, x_boundaries, y_boundaries):
    lower_x_boundary, upper_x_boundary = x_boundaries
    lower_y_boundary, upper_y_boundary = y_boundaries
    genome_length = 44
    half_genome = 22
    cromossomox = ""
    cromossomoy = ""
    population = []
    for i in range(pop_size):
        genome = generate_genome(genome_length)
        for i in range(len(genome)):
            if i < len(genome) / 2:
                cromossomox += str(genome[i])
                #print("x: " + cromossomox)
            else:
                cromossomoy += str(genome[i])
                #print("y: " + cromossomoy)

        individual_bit = {
            "x": cromossomox,
            "y": cromossomoy
        }

        cromossomox_int = int(bytearray(cromossomox, "utf8"),2)
        cromossomoy_int = int(bytearray(cromossomoy, "utf8"),2)

        #cromossomox_int = int(bytearray(cromossomox, "utf8"),2)
        #cromossomoy_int = int.from_bytes(bytearray(cromossomoy, "utf8"), byteorder='big', signed=False)

        '''
        print(cromossomox_int)
        sys.exit()
        '''

        cromossomox_real = Decimal(cromossomox_int) * Decimal(
                    (upper_x_boundary - lower_x_boundary) / (pow(2,half_genome) - 1)) + Decimal(lower_x_boundary)
        cromossomoy_real = Decimal(cromossomoy_int) * Decimal(
                    (upper_y_boundary - lower_y_boundary) / (pow(2,half_genome) - 1)) + Decimal(lower_y_boundary)

        individual = {
            "x": cromossomox_real,
            "y": cromossomoy_real
        }
        population.append(individual)
        cromossomox = ""
        cromossomoy = ""

    return population

def function_f6(x, y):
    x_y_pow = float(x ** 2 + y ** 2)
    square_of_pow = math.sqrt(x_y_pow)
    sin_pow = (math.sin(square_of_pow) ** 2)
    fraction = (((sin_pow) - (0.5))) / (((1 + (float(0.001) * (x_y_pow)))) ** 2)
    result = (0.5) - fraction
    return result

def apply_function(individual):
    x = individual["x"]
    y = individual["y"]
    return function_f6(x,y)


def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum
    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)
    draw = random.uniform(0, 1)
    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability
        if draw <= accumulated:
            return individual

def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)

#Alterar taxa de crossover
def crossover(individual_a, individual_b):
    xa = individual_a["x"]
    ya = individual_a["y"]
    xb = individual_b["x"]
    yb = individual_b["y"]

    return {"x": (xa + xb) / 2, "y": (ya + yb) / 2}

#Alterar taxa de Mutação
def mutate(individual):
    next_x = individual["x"] + random.uniform(-0.05, 0.05)
    next_y = individual["y"] + random.uniform(-0.05, 0.05)
    lower_boundary, upper_boundary = (-100, 100)
    # Guarantee we keep inside boundaries
    next_x = min(max(next_x, lower_boundary), upper_boundary)
    next_y = min(max(next_y, lower_boundary), upper_boundary)
    return {"x": next_x, "y": next_y}


def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in population)

    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual)
        next_generation.append(individual)

    return next_generation



generations = 40
population = generate_population(pop_size=100, x_boundaries=(-100, 100), y_boundaries=(-100, 100))
i = 1
Best_Values = []

# Falta realizar 20 experimentos
resultado = []
population_sorted = []

while True:
    print(f"🧬 GENERATION {i}")
    for individual in population:
        resultado = apply_function(individual)
        x = individual["x"]
        y = individual["y"]
        '''
        individual_sorted = {
            "x": x,
            "y":y,
            "resultado": resultado
        }
        population_sorted.append(individual_sorted)
        '''
    #population_sorted = sorted(population_sorted)

    if i == generations:
        break
    i += 1
    population = make_next_generation(population)

best_individual = sort_population_by_fitness(population)[-1]

#Falta acertar a media


print("\n🔬 Resultado Final 🔬")
print(best_individual, apply_function(best_individual))

