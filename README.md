# Optimal Pooling Strategies

This repository contains Python code developed for a main seminar at the **University of Stuttgart**. The topic of the seminar is **Optimal Pooling**, focusing on efficient testing strategies for large-scale population screening, particularly relevant during pandemics such as COVID-19.

## Seminar Topic: Optimal Pooling

Pooling strategies combine multiple samples into groups (or pools) to minimize the number of diagnostic tests required, especially in low-prevalence scenarios. This code explores and optimizes three main pooling strategies:
- **Fixed-size Pooling**
- **Hierarchical Pooling**
- **Adaptive Pooling**

The goal is to evaluate the performance of these strategies in terms of:
- Sensitivity
- Specificity
- Cost-effectiveness
- Total tests required

## Features

- **Population Simulation**: Generates a population with binary infection status (healthy or infected) based on a predefined prevalence.
- **Simulation of Pooling Strategies**:
  - Fixed-size pooling: Divides the population into pools of a fixed size.
  - Hierarchical pooling: Implements a multi-stage process where pools are subdivided if positive.
  - Adaptive pooling: Dynamically adjusts pool sizes based on the prevalence of infection.
- **Error Modeling**: Incorporates false positive and false negative rates into the simulations.
- **Optimization Algorithms**: Identifies the optimal pool size for each strategy to minimize total tests while maintaining accuracy.
- **Metrics Evaluation**:
  - Sensitivity
  - Specificity
  - False Negative Rate
  - Cost Effectiveness

## Usage

Clone the repository:
   ```bash
   git clone https://github.com/your-repo/optimal-pooling.git
   cd optimal-pooling
```
### Dependencies
The code requires the following dependencies:
- **Python 3.7+**
- **NumPy** (install using `pip install numpy`)

### Simulation Script
The main simulation script is `test_simulation.py`. It contains all the important functions used by the other analyses.


