from test_simulation import TestSimulation 

test_sim = TestSimulation()
# Simulation parameters
population_size = 10000
prevalence_rates = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, and 20% infection rates
pool_size = 10
false_positive_rate = 0.05
false_negative_rate = 0.1

# Evaluate the impact of prevalence on evaluation metrics
prevalence_metrics_results = test_sim.evaluate_prevalence_impact(prevalence_rates, population_size, pool_size, false_positive_rate, false_negative_rate)

# Display results
print("Impact of Prevalence Rates on Evaluation Metrics:")
for prevalence, strategies in prevalence_metrics_results.items():
    print(f"Prevalence: {prevalence*100:.1f}%")
    for strategy, metrics in strategies.items():
        print(f"  {strategy}: {metrics}")


