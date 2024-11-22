import numpy as np

def generate_population(population_size, prevalence):
    """Generate a population with binary infection status."""
    return np.random.choice([0, 1], size=population_size, p=[1-prevalence, prevalence])

def simulate_test(pool, false_positive_rate=0.01, false_negative_rate=0.05):
    """Simulate a test result for a pool of individuals, incorporating testing errors."""
    true_result = np.any(pool)  # True result: Is there at least one positive in the pool?
    
    if true_result:
        return np.random.choice([True, False], p=[1-false_negative_rate, false_negative_rate])
    else:
        return np.random.choice([False, True], p=[1-false_positive_rate, false_positive_rate])

def fixed_size_pooling(population, pool_size, false_positive_rate, false_negative_rate):
    """Simulate fixed-size pooling."""
    total_tests = 0
    tested_population = np.zeros_like(population)
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # One test per pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:  # If the pool tests positive
            total_tests += len(pool)  # Individual tests for all in the pool
            tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pool
    return total_tests, tested_population

def hierarchical_pooling(population, pool_size, false_positive_rate, false_negative_rate):
    """Simulate hierarchical pooling."""
    total_tests = 0
    tested_population = np.zeros_like(population)
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # First test the entire pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:  # If the pool tests positive
            sub_pools = [pool[j:j + (pool_size // 2)] for j in range(0, len(pool), pool_size // 2)]
            for sub_pool_idx, sub_pool in enumerate(sub_pools):
                total_tests += 1
                sub_pool_test_result = simulate_test(sub_pool, false_positive_rate, false_negative_rate)
                if sub_pool_test_result:
                    start_idx = pool_idx * pool_size + sub_pool_idx * (pool_size // 2)
                    tested_population[start_idx:start_idx + len(sub_pool)] = sub_pool
    return total_tests, tested_population

def adaptive_pooling(population, low_prevalence_pool_size, high_prevalence_pool_size, prevalence_threshold, false_positive_rate, false_negative_rate):
    """Simulate adaptive pooling."""
    total_tests = 0
    tested_population = np.zeros_like(population)
    estimated_prevalence = np.mean(population)
    pool_size = low_prevalence_pool_size if estimated_prevalence < prevalence_threshold else high_prevalence_pool_size
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # Test each pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:
            total_tests += len(pool)
            tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pool
    return total_tests, tested_population

def optimize_pooling_strategy(strategy_function, population, pool_size_range, false_positive_rate, false_negative_rate, **kwargs):
    """Optimize pooling size for a given strategy."""
    best_pool_size = None
    best_total_tests = float('inf')
    results = {}
    
    for pool_size in pool_size_range:
        if strategy_function == adaptive_pooling:
            total_tests, _ = strategy_function(population, pool_size, pool_size, **kwargs, false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)
        else:
            total_tests, _ = strategy_function(population, pool_size, false_positive_rate, false_negative_rate)
        results[pool_size] = total_tests
        if total_tests < best_total_tests:
            best_total_tests = total_tests
            best_pool_size = pool_size
    
    return best_pool_size, best_total_tests, results

def test_prevalence_rates(prevalence_rates, population_size, pool_size_range, false_positive_rate, false_negative_rate, prevalence_threshold=0.1):
    """
    Evaluate the performance of pooling strategies across different prevalence rates.
    """
    results = {}

    for prevalence in prevalence_rates:
        # Generate population for the given prevalence
        population = generate_population(population_size, prevalence)
        prevalence_results = {}

        # Optimize Fixed-size Pooling
        best_pool_fixed, best_tests_fixed, _ = optimize_pooling_strategy(
            fixed_size_pooling, population, pool_size_range, false_positive_rate, false_negative_rate
        )
        prevalence_results["Fixed-size Pooling"] = {"Optimal Pool Size": best_pool_fixed, "Total Tests": best_tests_fixed}

        # Optimize Hierarchical Pooling
        best_pool_hierarchical, best_tests_hierarchical, _ = optimize_pooling_strategy(
            hierarchical_pooling, population, pool_size_range, false_positive_rate, false_negative_rate
        )
        prevalence_results["Hierarchical Pooling"] = {"Optimal Pool Size": best_pool_hierarchical, "Total Tests": best_tests_hierarchical}

        # Optimize Adaptive Pooling
        best_pool_adaptive, best_tests_adaptive, _ = optimize_pooling_strategy(
            adaptive_pooling, population, pool_size_range, false_positive_rate, false_negative_rate,
            prevalence_threshold=prevalence_threshold
        )
        prevalence_results["Adaptive Pooling"] = {"Optimal Pool Size": best_pool_adaptive, "Total Tests": best_tests_adaptive}

        # Store results for this prevalence rate
        results[prevalence] = prevalence_results

    return results

# Simulation parameters
population_size = 100000
prevalence_rates = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, and 20% infection rates
false_positive_rate = 0.01
false_negative_rate = 0.05
pool_size_range = range(2, 200)  # Test pool sizes from 2 to 100
prevalence_threshold = 0.1

# Test pooling strategies across different prevalence rates
prevalence_results = test_prevalence_rates(prevalence_rates, population_size, pool_size_range, false_positive_rate, false_negative_rate)

# Display results
print("Results for Different Prevalence Rates:")
for prevalence, strategies in prevalence_results.items():
    print(f"\nPrevalence Rate: {prevalence*100:.0f}%")
    for strategy, metrics in strategies.items():
        print(f"  {strategy}: Optimal Pool Size = {metrics['Optimal Pool Size']}, Total Tests = {metrics['Total Tests']}")
