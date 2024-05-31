import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distances = self.calculate_distances()

    def calculate_distances(self):
        distances = np.random.randint(1, 100, size=(self.num_cities, self.num_cities))
        np.fill_diagonal(distances, 0)
        return distances

    def evaluate_fitness(self, solution):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distances[solution[i], solution[i+1]]
        total_distance += self.distances[solution[-1], solution[0]]
        return 1 / total_distance

class BatAlgorithm:
    def __init__(self, tsp, population_size, max_iterations, loudness, pulse_rate):
        self.tsp = tsp
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = 0
        self.best_distances = []
        self.best_solutions = []
        self.initial_city = 0
        self.best_iteration = 0
        self.best_distance = float('inf')
        self.shortest_distance_iteration = 0

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = np.random.permutation(self.tsp.num_cities)
            population.append(solution)
        return population

    def run(self):
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                new_solution = self.generate_new_solution(i)
                fitness = self.tsp.evaluate_fitness(new_solution)
                if np.random.rand() < self.pulse_rate[i] and fitness > self.tsp.evaluate_fitness(self.population[i]):
                    self.population[i] = new_solution
                    self.loudness[i] *= 0.9
                    self.pulse_rate[i] = 1 - np.exp(-0.9 * iteration)

                if fitness > self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = fitness
                    self.best_iteration = iteration
                    current_distance = 1 / fitness
                    if current_distance < self.best_distance: 
                        self.best_distance = current_distance
                        self.shortest_distance_iteration = iteration + 1

            self.best_distances.append(1 / self.best_fitness)
            self.best_solutions.append(self.best_solution)

    def generate_new_solution(self, i):
        new_solution = self.population[i].copy()
        j = np.random.randint(self.population_size)
        while j == i:
            j = np.random.randint(self.population_size)

        frequency = np.random.rand()
        new_solution = new_solution + (frequency * (self.population[i] - self.population[j]) * self.loudness[i]).astype(int)
        new_solution = np.clip(new_solution, 0, self.tsp.num_cities - 1)
        return new_solution

def animate(i):
    plt.clf()
    
    x = [city[0] for city in tsp.cities]
    y = [city[1] for city in tsp.cities]
    plt.scatter(x, y, c='b', s=50, label='Cities', zorder=1)
    
    route = bat_algorithm.best_solutions[i].tolist() + [bat_algorithm.best_solutions[i][0]]
    x_path = [tsp.cities[city][0] for city in route]
    y_path = [tsp.cities[city][1] for city in route]
    plt.plot(x_path, y_path, 'g-', linewidth=1, label='Path', zorder=2)
    
    for j, city in enumerate(route[:-1]):
        next_city = route[j+1]
        distance = tsp.distances[city, next_city]
        x_mid = (tsp.cities[city][0] + tsp.cities[next_city][0]) / 2
        y_mid = (tsp.cities[city][1] + tsp.cities[next_city][1]) / 2
        x_offset = (tsp.cities[next_city][0] - tsp.cities[city][0]) / 20
        y_offset = (tsp.cities[next_city][1] - tsp.cities[city][1]) / 20
        plt.text(x_mid + x_offset, y_mid + y_offset, str(distance), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5), zorder=3)

    init_city = bat_algorithm.initial_city
    plt.scatter(tsp.cities[init_city][0], tsp.cities[init_city][1], c='purple', s=100, marker='P', label='Initial City', zorder=4)
    
    salesman_city = route[0]
    plt.scatter(tsp.cities[salesman_city][0], tsp.cities[salesman_city][1], c='r', s=100, marker='*', label='Salesman', zorder=5)

    title = f"Iteration: {i+1}, Best Distance: {bat_algorithm.best_distances[i]:.2f}"
    
    current_shortest_distance = min(bat_algorithm.best_distances[:i+1])
    current_shortest_iteration = bat_algorithm.best_distances.index(current_shortest_distance) + 1
    title += f", Shortest Distance in Iteration {current_shortest_iteration}"
    
    plt.title(title)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    
def save_distance_matrix_plot(tsp):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(tsp.distances, cmap='viridis', interpolation='nearest')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(tsp.num_cities))
    ax.set_yticks(np.arange(tsp.num_cities))
    ax.set_xticklabels(np.arange(1, tsp.num_cities + 1))
    ax.set_yticklabels(np.arange(1, tsp.num_cities + 1))
    
    for i in range(tsp.num_cities):
        for j in range(tsp.num_cities):
            text = ax.text(j, i, tsp.distances[i, j], ha="center", va="center", color="w")
    
    ax.set_title("Distance Matrix")
    fig.tight_layout()
    plt.savefig('distance_matrix.png', dpi=300)
    plt.close()

def save_city_distances_plot(tsp):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    x = [city[0] for city in tsp.cities]
    y = [city[1] for city in tsp.cities]
    ax.scatter(x, y, c='b', s=50)
    
    for i in range(tsp.num_cities):
        for j in range(i+1, tsp.num_cities):
            x_mid = (tsp.cities[i][0] + tsp.cities[j][0]) / 2
            y_mid = (tsp.cities[i][1] + tsp.cities[j][1]) / 2
            ax.text(x_mid, y_mid, str(tsp.distances[i, j]), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    ax.set_title("City Distances")
    ax.axis('off')
    fig.tight_layout()
    plt.savefig('city_distances.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    num_cities_range = [20, 40, 60, 80, 100]
    num_bats_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_iterations = 100

    execution_times = np.zeros((len(num_bats_range), len(num_cities_range)))

    for i, num_bats in enumerate(num_bats_range):
        for j, num_cities in enumerate(num_cities_range):
            print(f"Running for {num_bats} bats and {num_cities} cities...")
            
            cities = np.random.rand(num_cities, 2) * 10
            tsp = TSP(cities)
            
            start_time = time.time()
            loudness = np.random.rand(num_bats)
            pulse_rate = np.random.rand(num_bats)
            bat_algorithm = BatAlgorithm(tsp, num_bats, max_iterations, loudness, pulse_rate)
            bat_algorithm.run()
            end_time = time.time()
            
            execution_times[i, j] = end_time - start_time

    plt.figure(figsize=(10, 6))
    for i, num_bats in enumerate(num_bats_range):
        plt.plot(num_cities_range, execution_times[i], marker='o', label=f"{num_bats} Bats")
    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Bat Algorithm TSP Time Complexity")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("bat_algorithm_tsp_time_complexity.png", dpi=300)
    plt.show()