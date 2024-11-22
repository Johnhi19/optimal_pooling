import numpy as np

def generate_population(population_size, prevalence):
    """
    Generate a population with binary infection status.
    1 represents an infected individual, and 0 represents a healthy individual.
    """
    return np.random.choice([0, 1], size=population_size, p=[1-prevalence, prevalence])

def simulate_test(pool, false_positive_rate=0.01, false_negative_rate=0.05):
    """
    Simulate a test result for a pool of individuals, incorporating testing errors.
    - False Positive Rate: Probability a healthy pool tests positive.
    - False Negative Rate: Probability an infected pool tests negative.
    """
    true_result = np.any(pool)  # True result: Is there at least one positive in the pool?
    
    if true_result:
        # False negative: pool tests negative despite having infections
        return np.random.choice([True, False], p=[1-false_negative_rate, false_negative_rate])
    else:
        # False positive: pool tests positive despite being completely healthy
        return np.random.choice([False, True], p=[1-false_positive_rate, false_positive_rate])

def evaluate_metrics(true_population, tested_population, total_tests):
    """
    Calculate sensitivity, specificity, false negative rate, and cost effectiveness.
    """
    true_positives = np.sum((true_population == 1) & (tested_population == 1))
    false_negatives = np.sum((true_population == 1) & (tested_population == 0))
    true_negatives = np.sum((true_population == 0) & (tested_population == 0))
    false_positives = np.sum((true_population == 0) & (tested_population == 1))

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    cost_effectiveness = total_tests / len(true_population)

    return sensitivity, specificity, false_negative_rate, cost_effectiveness

def individual_testing(population, false_positive_rate, false_negative_rate):
    """
    Simulate individual testing.
    Each individual is tested separately, with potential for false positives and negatives.
    """
    total_tests = len(population)
    tested_population = np.array([simulate_test([ind], false_positive_rate, false_negative_rate) for ind in population])
    return total_tests, tested_population

def fixed_size_pooling(population, pool_size, false_positive_rate, false_negative_rate):
    """
    Simulate fixed-size pooling.
    Divide the population into pools of fixed size and test each pool.
    """
    total_tests = 0
    tested_population = np.zeros_like(population)
    
    # Split population into pools of fixed size
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # One test per pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:  # If the pool tests positive
            total_tests += len(pool)  # Individual tests for all in the pool
            tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pool
    
    return total_tests, tested_population

def hierarchical_pooling(population, pool_size, false_positive_rate, false_negative_rate):
    """
    Simulate hierarchical pooling.
    Start with a large pool, and test smaller sub-pools if the large pool tests positive.
    """
    total_tests = 0
    tested_population = np.zeros_like(population)
    
    # Split population into pools of fixed size
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # First test the entire pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:  # If the pool tests positive
            sub_pools = [pool[j:j + (pool_size // 2)] for j in range(0, len(pool), pool_size // 2)]
            for sub_pool_idx, sub_pool in enumerate(sub_pools):
                total_tests += 1
                sub_pool_test_result = simulate_test(sub_pool, false_positive_rate, false_negative_rate)
                if sub_pool_test_result:  # If a sub-pool tests positive
                    start_idx = pool_idx * pool_size + sub_pool_idx * (pool_size // 2)
                    tested_population[start_idx:start_idx + len(sub_pool)] = sub_pool
    
    return total_tests, tested_population

def adaptive_pooling(population, low_prevalence_pool_size, high_prevalence_pool_size, prevalence_threshold, false_positive_rate, false_negative_rate):
    """
    Simulate adaptive pooling.
    Adjust pool sizes dynamically based on prevalence estimates.
    """
    total_tests = 0
    tested_population = np.zeros_like(population)
    estimated_prevalence = np.mean(population)  # Initial prevalence estimate
    
    if estimated_prevalence < prevalence_threshold:
        pool_size = low_prevalence_pool_size
    else:
        pool_size = high_prevalence_pool_size
    
    # Split population into adaptive pools
    pools = [population[i:i + pool_size] for i in range(0, len(population), pool_size)]
    for pool_idx, pool in enumerate(pools):
        total_tests += 1  # Test each pool
        pool_test_result = simulate_test(pool, false_positive_rate, false_negative_rate)
        if pool_test_result:  # If a pool tests positive
            total_tests += len(pool)
            tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pool
    
    return total_tests, tested_population

# Simulation parameters
population_size = 1000000
prevalence = 0.05  # 5% infection rate
pool_size = 10
low_prevalence_pool_size = 15
high_prevalence_pool_size = 5
prevalence_threshold = 0.1
false_positive_rate = 0.01
false_negative_rate = 0.05

# Generate a population
population = generate_population(population_size, prevalence)

# Simulate each strategy
results = {}

# Individual Testing
total_tests, tested_population = individual_testing(population, false_positive_rate, false_negative_rate)
results['Individual Testing'] = evaluate_metrics(population, tested_population, total_tests)

# Fixed-size Pooling
total_tests, tested_population = fixed_size_pooling(population, pool_size, false_positive_rate, false_negative_rate)
results['Fixed-size Pooling'] = evaluate_metrics(population, tested_population, total_tests)

# Hierarchical Pooling
total_tests, tested_population = hierarchical_pooling(population, pool_size, false_positive_rate, false_negative_rate)
results['Hierarchical Pooling'] = evaluate_metrics(population, tested_population, total_tests)

# Adaptive Pooling
total_tests, tested_population = adaptive_pooling(population, low_prevalence_pool_size, high_prevalence_pool_size, prevalence_threshold, false_positive_rate, false_negative_rate)
results['Adaptive Pooling'] = evaluate_metrics(population, tested_population, total_tests)

# Display results
for strategy, (sensitivity, specificity, fnr, cost_effectiveness) in results.items():
    print(f"{strategy}:")
    print(f"  Sensitivity: {sensitivity:.2f}")
    print(f"  Specificity: {specificity:.2f}")
    print(f"  False Negative Rate: {fnr:.2f}")
    print(f"  Cost Effectiveness (Tests per Person): {cost_effectiveness:.2f}")
    print()
