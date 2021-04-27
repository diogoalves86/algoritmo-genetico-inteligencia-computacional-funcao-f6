import random
import math
import statistics


def generate_population(size, x_boundaries, y_boundaries):
    lower_x_boundary, upper_x_boundary = x_boundaries
    lower_y_boundary, upper_y_boundary = y_boundaries

    population = []
    for i in range(size):
        individual = {
            "x": random.uniform(lower_x_boundary, upper_x_boundary),
            "y": random.uniform(lower_y_boundary, upper_y_boundary),
        }
        population.append(individual)

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
    lower_boundary, upper_boundary = (-4, 4)
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

population = generate_population(size=100, x_boundaries=(-100, 100), y_boundaries=(-100, 100))

i = 1
Best_Values = []
# Falta realizar 20 experimentos
while True:
    print(f"🧬 GENERATION {i}")
    for individual in population:
        print(individual, apply_function(individual))
    if i == generations:
        break
    i += 1

    population = make_next_generation(population)

best_individual = sort_population_by_fitness(population)[-1]

#Falta acertar a media


print("\n🔬 Resultado Final 🔬")
print(best_individual, apply_function(best_individual))

