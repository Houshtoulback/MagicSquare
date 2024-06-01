import random
import math
import copy
import matplotlib.pyplot as plt


class MagicSquare:
    def __init__(self, chrom):
        self.chrom = chrom
        self.fitness = calculate_fitness(chrom)

    def __str__(self):
        return f"chromosome {self.chrom} fitness: {self.fitness}"


def generate_initial_population(population_size, n):
    population = []
    while len(population) < population_size:
        square = random.sample(range(1, n ** 2 + 1), n ** 2)
        if square not in population:
            population.append(MagicSquare(square))

    sorted_population = sorted(population, key=lambda obj: obj.fitness)
    return sorted_population


def calculate_fitness(square):
    n = int(math.sqrt(len(square)))
    magic_number = n * (n ** 2 + 1) // 2
    rows = [square[i:i + n] for i in range(0, n ** 2, n)]
    cols = [square[i::n] for i in range(n)]
    if n > 1:
        diag1 = [square[i] for i in range(0, n ** 2, n + 1)]
        diag2 = [square[i] for i in range(n - 1, n ** 2 - 1, n - 1)]
    else:  # When n is 1, diag2 will be just the single element in the square
        diag1 = [square[0]]
        diag2 = [square[0]]

    row_sums = [sum(row) for row in rows]
    col_sums = [sum(col) for col in cols]
    diag1_sum = sum(diag1)
    diag2_sum = sum(diag2)

    fitness = sum([abs(magic_number - val) for val in row_sums + col_sums + [diag1_sum, diag2_sum]])

    return fitness


def pmx_crossover(parent1, parent2, crossover_rate):
    assert len(parent1) == len(parent2), "Both parents must be of the same length."

    length = len(parent1)
    offspring1 = parent1[:]
    offspring2 = parent2[:]

    if random.random() < crossover_rate:
        # Choose two random crossover points
        cx_point1 = random.randint(0, length - 1)
        cx_point2 = random.randint(0, length - 1)

        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1

        # Apply PMX crossover
        def pmx(parent1, parent2, offspring):
            mapping = {}
            for i in range(cx_point1, cx_point2 + 1):
                offspring[i] = parent2[i]
                mapping[parent2[i]] = parent1[i]

            for i in range(length):
                if i < cx_point1 or i > cx_point2:
                    while offspring[i] in mapping:
                        offspring[i] = mapping[offspring[i]]

        pmx(parent1, parent2, offspring1)
        pmx(parent2, parent1, offspring2)

    return offspring1, offspring2


def roulette_wheel_selection(population):
    fitness_scores = [obj.fitness for obj in population]
    total_fitness = sum(fitness_scores)
    probabilities = [(total_fitness - score) / total_fitness for score in fitness_scores]
    selected = random.choices(population, probabilities, k=2)
    return selected


def mutation(square, mutation_rate):
    # Randomly select two distinct indices
    n = int(math.sqrt(len(square)))
    if random.random() < mutation_rate:
        index1 = random.randint(0, n ** 2 - 1)
        index2 = random.randint(0, n ** 2 - 1)
        while index2 == index1:
            index2 = random.randint(0, n ** 2 - 1)

        # Swap the values at the chosen indices
        square[index1], square[index2] = square[index2], square[index1]
    return square


def scramble_mutation(array, mutation_rate):
    if random.random() > mutation_rate:
        return array  # No mutation occurs

    length = len(array)
    num_elements_to_swap = max(1, int(math.ceil(length * 0.05)))  # Ensure at least one element is swapped

    mutated_array = array[:]

    for _ in range(num_elements_to_swap):
        i, j = random.sample(range(length), 2)  # Select two distinct indices to swap
        mutated_array[i], mutated_array[j] = mutated_array[j], mutated_array[i]

    return mutated_array


def evolve(population, population_size, crossover_rate, mutation_rate, elitism_rate):
    next_population = []
    while len(next_population) < (1 - elitism_rate) * population_size:
        parent1, parent2 = roulette_wheel_selection(population)
        offspring1, offspring2 = pmx_crossover(parent1.chrom, parent2.chrom, crossover_rate)
        mutated_offspring1 = scramble_mutation(offspring1, mutation_rate)
        mutated_offspring2 = scramble_mutation(offspring2, mutation_rate)
        next_population.append(MagicSquare(mutated_offspring1))
        next_population.append(MagicSquare(mutated_offspring2))

    while len(next_population) < population_size:
        next_population.append(population.pop(0))
    next_population = sorted(next_population, key=lambda chrom: chrom.fitness)
    return next_population


def run_genetic(square_size, population_size, generation_num, crossover_rate, mutation_rate, elitism_rate):
    fitness_scores = []
    population = generate_initial_population(population_size, square_size)
    for generation in range(generation_num):
        population = evolve(population, population_size, crossover_rate, mutation_rate, elitism_rate)
        # print(f"generation: number{generation + 1}")
        fitness_scores.append(population[0].fitness)
        if population[0].fitness == 0:
            print(f"the answer: {population[0]}")
            plt.plot(range(1, generation + 2), fitness_scores)
            plt.xlabel("generation")
            plt.ylabel("fitness")
            plt.title("best fitness over generations: ")
            plt.show()
            return population[0]
        # print("the best chromosome: ", population[0])
        if generation == generation_num - 1:
            plt.plot(range(1, generation + 2), fitness_scores)
            plt.xlabel("generation")
            plt.ylabel("fitness")
            plt.title("best fitness over generations: ")
            plt.show()

    return 0


run_genetic(9, 2000, 2000, 1, 0.1, 0.05)
