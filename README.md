# TSP Solver

A Python implementation of the Traveling Salesman Problem (TSP) featuring multiple algorithms and visualization.

## Features

- **Adjacency Matrix** representation for graph distances
- **Three solving algorithms:**
  - Brute Force (exact optimal solution)
  - Nearest Neighbor (greedy heuristic)
  - 2-Opt (local search improvement)
- **Visualization** with matplotlib showing tour direction
- **Performance comparison** table with timing and solution quality

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- numpy
- matplotlib

## Usage

### Run the Demo

```bash
python tsp_solver.py
```

This generates an 8-city problem, solves it with all algorithms, and displays the best tour.

### Use as a Module

```python
from tsp_solver import (
    generate_random_cities,
    create_distance_matrix,
    brute_force_tsp,
    nearest_neighbor_best,
    two_opt_tsp,
    visualize_tour
)

# Generate cities
cities = generate_random_cities(8, seed=42)

# Create distance matrix
matrix = create_distance_matrix(cities)

# Solve with different algorithms
bf_tour, bf_dist = brute_force_tsp(matrix)
nn_tour, nn_dist = nearest_neighbor_best(matrix)
opt_tour, opt_dist = two_opt_tsp(matrix)

# Visualize
visualize_tour(cities, bf_tour, bf_dist)
```

## Algorithms

| Algorithm | Time Complexity | Optimal? | Best For |
|-----------|-----------------|----------|----------|
| Brute Force | O((n-1)!) | ✓ Yes | n ≤ 10 cities |
| Nearest Neighbor | O(n²) | ✗ No | Quick approximation |
| 2-Opt | O(n² × iter) | ✗ No | Improving existing tours |

## Example Output

```
===========================================================================
           TSP SOLVER - Complete Demo
===========================================================================

Generated 8 random cities (seed=42):
  City 0: ( 63.94,   2.53)
  City 1: ( 27.50,  22.32)
  ...

Algorithm          Tour                     Distance       Time      Gap
---------------------------------------------------------------------------
Brute Force        [0, 2, 1, 6, 4, 7, 5, 3]     247.35    12.34ms  optimal
Nearest Neighbor   [0, 3, 5, 7, 4, 6, 1, 2]     261.82     0.08ms   +5.8%
2-Opt              [0, 2, 1, 6, 4, 7, 5, 3]     247.35     0.21ms  optimal
---------------------------------------------------------------------------

✓ Best solution: Brute Force
  Distance: 247.35
```

## File Structure

```
TSP/
├── tsp_solver.py      # Main solver with all algorithms
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── tsp_tour.png       # Generated visualization (after running)
```

## License

MIT

