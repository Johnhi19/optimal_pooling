import numpy as np

class TestSimulation:

    def generate_population(self, population_size, prevalence):
        """
        Generate a population with binary infection status.
        1 represents an infected individual, and 0 represents a healthy individual.
        """
        return np.random.choice([0, 1], size=population_size, p=[1-prevalence, prevalence])

    def simulate_test(self, pool, false_positive_rate=0.01, false_negative_rate=0.05):
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

    def evaluate_metrics(self, true_population, tested_population, total_tests):
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

    def individual_testing(self, population, false_positive_rate, false_negative_rate):
        """
        Simulate individual testing.
        Each individual is tested separately, with potential for false positives and negatives.
        """
        total_tests = len(population)
        tested_population = np.array([self.simulate_test([ind], false_positive_rate, false_negative_rate) for ind in population])
        return total_tests, tested_population

    def fixed_size_pooling(self, population, pool_size, false_positive_rate, false_negative_rate):
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
            pool_test_result = self.simulate_test(pool, false_positive_rate, false_negative_rate)
            if pool_test_result:  # If the pool tests positive
                total_tests += len(pool)  # Individual tests for all in the pool
                number_tests, pop = self.individual_testing(pool, false_positive_rate, false_negative_rate)
                tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pop
        
        return total_tests, tested_population

    def hierarchical_pooling(self, population, pool_size, false_positive_rate, false_negative_rate):
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
            pool_test_result = self.simulate_test(pool, false_positive_rate, false_negative_rate)
            if pool_test_result:  # If the pool tests positive
                sub_pools = [pool[j:j + (pool_size // 2)] for j in range(0, len(pool), pool_size // 2)]
                for sub_pool_idx, sub_pool in enumerate(sub_pools):
                    total_tests += 1
                    sub_pool_test_result = self.simulate_test(sub_pool, false_positive_rate, false_negative_rate)
                    if sub_pool_test_result:  # If a sub-pool tests positive
                        start_idx = pool_idx * pool_size + sub_pool_idx * (pool_size // 2)
                        number_tests, pop = self.individual_testing(pool, false_positive_rate, false_negative_rate)
                        tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pop
        
        return total_tests, tested_population

    def adaptive_pooling(self, population, low_prevalence_pool_size, high_prevalence_pool_size, prevalence_threshold, false_positive_rate, false_negative_rate):
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
            pool_test_result = self.simulate_test(pool, false_positive_rate, false_negative_rate)
            if pool_test_result:  # If a pool tests positive
                total_tests += len(pool)
                number_tests, pop = self.individual_testing(pool, false_positive_rate, false_negative_rate)
                tested_population[pool_idx * pool_size:pool_idx * pool_size + len(pool)] = pop
        
        return total_tests, tested_population

    def evaluate_prevalence_impact(self, prevalence_rates, population_size, pool_size, 
                        false_positive_rate, false_negative_rate, prevalence_threshold=0.1, num_simulations=10):
        """
        Simulate pooling strategies multiple times for each prevalence rate and calculate average metrics.
        
        Parameters:
        - prevalence_rates: List of prevalence rates to evaluate.
        - population_size: Size of the simulated population.
        - pool_size: Fixed pool size for strategies.
        - prevalence_threshold: Threshold for adaptive pooling to switch between low and high prevalence pool sizes.
        - false_positive_rate: Probability of a false positive.
        - false_negative_rate: Probability of a false negative.
        - num_simulations: Number of times each simulation is repeated.
        
        Returns:
        - Dictionary with averaged metrics for each prevalence rate and pooling strategy.
        """
        results = {}
        
        for prevalence in prevalence_rates:
            prevalence_results = {
                "Fixed-size Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
                "Hierarchical Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
                "Adaptive Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
            }
            
            for _ in range(num_simulations):
                # Generate population for the given prevalence
                population = self.generate_population(population_size, prevalence)
                
                # Fixed-size Pooling
                total_tests, tested_population = self.fixed_size_pooling(population, pool_size, false_positive_rate, false_negative_rate)
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Fixed-size Pooling"][key].append(value)
                
                # Hierarchical Pooling
                total_tests, tested_population = self.hierarchical_pooling(population, pool_size, false_positive_rate, false_negative_rate)
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Hierarchical Pooling"][key].append(value)
                
                # Adaptive Pooling
                total_tests, tested_population = self.adaptive_pooling(
                    population, 
                    low_prevalence_pool_size=pool_size, 
                    high_prevalence_pool_size=pool_size, 
                    prevalence_threshold=prevalence_threshold, 
                    false_positive_rate=false_positive_rate, 
                    false_negative_rate=false_negative_rate
                )
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Adaptive Pooling"][key].append(value)
            
            # Average the metrics for this prevalence rate
            results[prevalence] = {
                strategy: {
                    metric: np.mean(values) for metric, values in strategy_metrics.items()
                }
                for strategy, strategy_metrics in prevalence_results.items()
            }
        
        return results
    
    def optimize_pooling_strategy(self, strategy_function, population, pool_size_range, false_positive_rate, false_negative_rate, **kwargs):
        """Optimize pooling size for a given strategy."""
        best_pool_size = None
        best_total_tests = float('inf')
        results = {}
        
        for pool_size in pool_size_range:
            if strategy_function == self.adaptive_pooling:
                total_tests, _ = strategy_function(population, pool_size, pool_size, **kwargs, false_positive_rate=false_positive_rate, false_negative_rate=false_negative_rate)
            else:
                total_tests, _ = strategy_function(population, pool_size, false_positive_rate, false_negative_rate)
            results[pool_size] = total_tests
            if total_tests < best_total_tests:
                best_total_tests = total_tests
                best_pool_size = pool_size
        
        return best_pool_size, best_total_tests, results

    def evaluate_prevalence_impact(self, prevalence_rates, population_size, pool_size, 
                        false_positive_rate, false_negative_rate, prevalence_threshold=0.1, num_simulations=10):
        """
        Simulate pooling strategies multiple times for each prevalence rate and calculate average metrics.
        
        Parameters:
        - prevalence_rates: List of prevalence rates to evaluate.
        - population_size: Size of the simulated population.
        - pool_size: Fixed pool size for strategies.
        - prevalence_threshold: Threshold for adaptive pooling to switch between low and high prevalence pool sizes.
        - false_positive_rate: Probability of a false positive.
        - false_negative_rate: Probability of a false negative.
        - num_simulations: Number of times each simulation is repeated.
        
        Returns:
        - Dictionary with averaged metrics for each prevalence rate and pooling strategy.
        """
        results = {}
        
        for prevalence in prevalence_rates:
            prevalence_results = {
                "Individual Testing": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
                "Fixed-size Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
                "Hierarchical Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
                "Adaptive Pooling": {"sensitivity": [], "specificity": [], "fnr": [], "cost_effectiveness": []},
            }
            
            for _ in range(num_simulations):
                # Generate population for the given prevalence
                population = self.generate_population(population_size, prevalence)

                # Individual Testing
                total_tests, tested_population = self.individual_testing(population, false_positive_rate, false_negative_rate)
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Individual Testing"][key].append(value)
                
                # Fixed-size Pooling
                total_tests, tested_population = self.fixed_size_pooling(population, pool_size, false_positive_rate, false_negative_rate)
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Fixed-size Pooling"][key].append(value)
                
                # Hierarchical Pooling
                total_tests, tested_population = self.hierarchical_pooling(population, pool_size, false_positive_rate, false_negative_rate)
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Hierarchical Pooling"][key].append(value)
                
                # Adaptive Pooling
                total_tests, tested_population = self.adaptive_pooling(
                    population, 
                    low_prevalence_pool_size=pool_size, 
                    high_prevalence_pool_size=pool_size, 
                    prevalence_threshold=prevalence_threshold, 
                    false_positive_rate=false_positive_rate, 
                    false_negative_rate=false_negative_rate
                )
                metrics = self.evaluate_metrics(population, tested_population, total_tests)
                for key, value in zip(["sensitivity", "specificity", "fnr", "cost_effectiveness"], metrics):
                    prevalence_results["Adaptive Pooling"][key].append(value)
            
            # Average the metrics for this prevalence rate
            results[prevalence] = {
                strategy: {
                    metric: np.mean(values) for metric, values in strategy_metrics.items()
                }
                for strategy, strategy_metrics in prevalence_results.items()
            }
        
        return results