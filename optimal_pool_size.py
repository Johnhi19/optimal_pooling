from test_simulation import TestSimulation 


# Simulation parameters
population_size = 100000
prevalence = 0.05
false_positive_rate = 0.05
false_negative_rate = 0.1
pool_size_range = range(2, 100)
prevalence_threshold = 0.1

test_sim = TestSimulation()
# Generate a population
population = test_sim.generate_population(population_size, prevalence)

# Optimize Fixed-size Pooling
best_pool_fixed, best_tests_fixed, results_fixed = test_sim.optimize_pooling_strategy(
    test_sim.fixed_size_pooling, population, pool_size_range, false_positive_rate, false_negative_rate
)

# Optimize Hierarchical Pooling
best_pool_hierarchical, best_tests_hierarchical, results_hierarchical = test_sim.optimize_pooling_strategy(
    test_sim.hierarchical_pooling, population, pool_size_range, false_positive_rate, false_negative_rate
)

# Optimize Adaptive Pooling
best_pool_adaptive, best_tests_adaptive, results_adaptive = test_sim.optimize_pooling_strategy(
    test_sim.adaptive_pooling, population, pool_size_range, false_positive_rate, false_negative_rate,
    prevalence_threshold=prevalence_threshold
)

# Display optimization results
print(f"Fixed-size Pooling: Optimal Pool Size = {best_pool_fixed}, Total Tests = {best_tests_fixed}")
print(f"Hierarchical Pooling: Optimal Pool Size = {best_pool_hierarchical}, Total Tests = {best_tests_hierarchical}")
print(f"Adaptive Pooling: Optimal Pool Size = {best_pool_adaptive}, Total Tests = {best_tests_adaptive}")

# # Detailed results for each pooling strategy
print("\nDetailed Results by Pool Size:")
print("Fixed-size Pooling:")
for pool_size, total_tests in results_fixed.items():
    print(f"  Pool Size {pool_size}: Total Tests = {total_tests}")

print("\nHierarchical Pooling:")
for pool_size, total_tests in results_hierarchical.items():
    print(f"  Pool Size {pool_size}: Total Tests = {total_tests}")

print("\nAdaptive Pooling:")
for pool_size, total_tests in results_adaptive.items():
    print(f"  Pool Size {pool_size}: Total Tests = {total_tests}")
