import numpy as np

def genetic_algorithm_extended(objective_func, bounds, pop_size=100, max_generations=1000,
                               mutation_rate=0.5, crossover_rate=1.0, elitism=True,
                               tournament_size=5, seed=None):
    np.random.seed(seed)
    n_params = len(bounds)
    
    # Khởi tạo quần thể
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, n_params))
    
    # Hàm tính fitness cho một cá thể
    def fitness(individual):
        return objective_func(individual)
    
    # Chọn lọc bằng tournament selection
    def tournament_selection(population, fitness_values):
        selected = np.random.choice(len(population), size=tournament_size, replace=False)
        selected_fitness = fitness_values[selected]
        winner = selected[np.argmin(selected_fitness)]  # Chọn cá thể có fitness tốt nhất
        return population[winner]
    
    # Two-point crossover
    def crossover(parents):
        crossover_point1 = np.random.randint(1, n_params-1)
        crossover_point2 = np.random.randint(crossover_point1, n_params)
        offspring = np.copy(parents)
        offspring[0, crossover_point1:crossover_point2] = parents[1, crossover_point1:crossover_point2]
        offspring[1, crossover_point1:crossover_point2] = parents[0, crossover_point1:crossover_point2]
        return offspring
    
    # Đột biến Gaussian
    def mutate(individual, mutation_rate):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, n_params)
            mutation_value = np.random.normal(0, 1)  # Gaussian mutation
            individual[mutation_point] += mutation_value
            individual[mutation_point] = np.clip(individual[mutation_point], bounds[mutation_point, 0], bounds[mutation_point, 1])
        return individual
    
    best_solution = None
    best_fitness = np.inf

    # Tiến hành qua các thế hệ
    for generation in range(max_generations):
        fitness_values = np.array([fitness(ind) for ind in population])
        
        # Cập nhật giải pháp tốt nhất
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitness_values)]
        
        # Elitism: Giữ lại cá thể tốt nhất từ thế hệ trước
        if elitism:
            elite_solution = best_solution
        
        # Quá trình chọn lọc, lai ghép và đột biến
        new_population = []
        for _ in range(pop_size // 2):  # tạo quần thể mới từ các cặp cha mẹ
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            offspring = crossover(np.array([parent1, parent2]))
            offspring[0] = mutate(offspring[0], mutation_rate)
            offspring[1] = mutate(offspring[1], mutation_rate)
            new_population.extend([offspring[0], offspring[1]])
        
        # Nếu sử dụng elitism, thêm cá thể tốt nhất vào quần thể
        if elitism:
            new_population[0] = elite_solution  # Thêm giải pháp tốt nhất vào vị trí đầu tiên
        
        population = np.array(new_population)
        
    return best_solution, best_fitness

# Ví dụ về hàm mục tiêu (Sphere Function)
def sphere_function(x):
    return np.sum(x**2)

# Thiết lập phạm vi cho bài toán
bounds = np.array([[-5.12, 5.12]]*10)

# Chạy thuật toán di truyền mở rộng
best_solution, best_fitness = genetic_algorithm_extended(sphere_function, bounds)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
