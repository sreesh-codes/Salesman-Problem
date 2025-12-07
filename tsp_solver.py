"""
TSP Solver - Traveling Salesman Problem Implementation
Algorithms: Brute Force, Nearest Neighbor, 2-Opt with Visualization
"""

import itertools
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# SECTION 1: Graph Representation (Adjacency Matrix)
# =============================================================================

def create_distance_matrix(cities: List[Tuple[float, float]]) -> np.ndarray:
    """Create adjacency matrix from city coordinates using Euclidean distance."""
    n = len(cities)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx, dy = cities[i][0] - cities[j][0], cities[i][1] - cities[j][1]
                matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    return matrix


def generate_random_cities(n: int, seed: Optional[int] = None) -> List[Tuple[float, float]]:
    """Generate n random cities with coordinates in [0, 100] range."""
    if seed is not None:
        random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


def calculate_tour_distance(tour: List[int], matrix: np.ndarray) -> float:
    """Calculate total distance of a tour (returns to start)."""
    return sum(matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


# =============================================================================
# SECTION 2: Brute Force Algorithm (Exact Solution)
# =============================================================================

def brute_force_tsp(matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Find optimal tour by checking all permutations.
    Time: O((n-1)!) - Only practical for n <= 10
    """
    n = len(matrix)
    if n <= 1:
        return list(range(n)), 0.0
    
    best_tour, best_dist = None, float('inf')
    
    # Fix first city to reduce search space (eliminates symmetric tours)
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        dist = calculate_tour_distance(tour, matrix)
        if dist < best_dist:
            best_tour, best_dist = tour, dist
    
    return best_tour, best_dist


# =============================================================================
# SECTION 3: Nearest Neighbor Algorithm (Greedy Heuristic)
# =============================================================================

def nearest_neighbor_tsp(matrix: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Greedy heuristic: always visit the nearest unvisited city.
    Time: O(n²)
    """
    n = len(matrix)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    
    for _ in range(n - 1):
        # Find nearest unvisited city (greedy choice)
        nearest, nearest_dist = None, float('inf')
        for j in range(n):
            if not visited[j] and matrix[current][j] < nearest_dist:
                nearest, nearest_dist = j, matrix[current][j]
        
        if nearest is None:
            break  # Graph disconnected
        
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    
    return tour, calculate_tour_distance(tour, matrix)


def nearest_neighbor_best(matrix: np.ndarray) -> Tuple[List[int], float]:
    """Run nearest neighbor from all starting points, return best."""
    best_tour, best_dist = None, float('inf')
    for start in range(len(matrix)):
        tour, dist = nearest_neighbor_tsp(matrix, start)
        if dist < best_dist:
            best_tour, best_dist = tour, dist
    return best_tour, best_dist


# =============================================================================
# SECTION 4: 2-Opt Algorithm (Local Search Improvement)
# =============================================================================

def two_opt_swap(tour: List[int], i: int, j: int) -> List[int]:
    """Reverse segment between indices i and j (2-opt move)."""
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]


def two_opt_tsp(matrix: np.ndarray, initial_tour: Optional[List[int]] = None,
                max_iter: int = 1000) -> Tuple[List[int], float]:
    """
    Improve tour by swapping edges that reduce distance.
    Time: O(n² × iterations)
    """
    n = len(matrix)
    
    # Start with nearest neighbor if no initial tour
    if initial_tour is None:
        tour, _ = nearest_neighbor_best(matrix)
    else:
        tour = initial_tour.copy()
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate improvement without full tour recalculation
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                old_cost = matrix[a][b] + matrix[c][d]
                new_cost = matrix[a][c] + matrix[b][d]
                
                if new_cost < old_cost - 1e-10:
                    tour = two_opt_swap(tour, i, j)
                    improved = True
                    break
            if improved:
                break
    
    return tour, calculate_tour_distance(tour, matrix)


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_tour(cities: List[Tuple[float, float]], tour: List[int],
                   distance: Optional[float] = None) -> None:
    """Plot TSP tour with arrows showing direction."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Draw tour path with arrows
    for k in range(len(tour)):
        i, j = tour[k], tour[(k + 1) % len(tour)]
        ax.annotate('', xy=cities[j], xytext=cities[i],
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))
    
    # Plot cities
    xs, ys = zip(*cities)
    ax.scatter(xs, ys, s=120, c='#E94F37', edgecolors='white', linewidths=2, zorder=3)
    
    # Highlight start (green) and end (orange)
    ax.scatter(*cities[tour[0]], s=200, c='#44AF69', edgecolors='white', 
               linewidths=2, zorder=4, label='Start')
    ax.scatter(*cities[tour[-1]], s=200, c='#F4A261', edgecolors='white', 
               linewidths=2, zorder=4, label='End')
    
    # Label cities
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontweight='bold')
    
    # Title with distance
    title = f"TSP Tour ({len(cities)} cities)"
    if distance is not None:
        title += f" — Distance: {distance:.2f}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('tsp_tour.png', dpi=150)
    plt.show()


# =============================================================================
# SECTION 6: Algorithm Comparison
# =============================================================================

def compare_algorithms(matrix: np.ndarray) -> dict:
    """Run all algorithms and return results with timing."""
    results = {}
    
    # Brute Force
    t0 = time.perf_counter()
    tour, dist = brute_force_tsp(matrix)
    results['Brute Force'] = (tour, dist, (time.perf_counter() - t0) * 1000)
    
    # Nearest Neighbor
    t0 = time.perf_counter()
    tour, dist = nearest_neighbor_best(matrix)
    results['Nearest Neighbor'] = (tour, dist, (time.perf_counter() - t0) * 1000)
    
    # 2-Opt
    t0 = time.perf_counter()
    tour, dist = two_opt_tsp(matrix)
    results['2-Opt'] = (tour, dist, (time.perf_counter() - t0) * 1000)
    
    return results


def print_comparison_table(results: dict) -> None:
    """Print formatted comparison table of algorithm results."""
    optimal = results['Brute Force'][1]
    
    print(f"\n{'Algorithm':<18} {'Tour':<24} {'Distance':>10} {'Time':>10} {'Gap':>8}")
    print("-" * 75)
    
    for name, (tour, dist, ms) in results.items():
        tour_str = str(tour) if len(str(tour)) <= 22 else str(tour[:3])[:-1] + '...]'
        gap = ((dist - optimal) / optimal) * 100
        gap_str = "optimal" if gap < 0.01 else f"+{gap:.1f}%"
        print(f"{name:<18} {tour_str:<24} {dist:>10.2f} {ms:>8.2f}ms {gap_str:>8}")


# =============================================================================
# SECTION 7: Main Demo
# =============================================================================

def main():
    """Run complete TSP demo: solve example problem with all algorithms."""
    print("\n" + "=" * 75)
    print("           TSP SOLVER - Complete Demo")
    print("=" * 75)
    
    # Generate 8-city problem
    n_cities = 8
    seed = 42
    cities = generate_random_cities(n_cities, seed=seed)
    
    print(f"\nGenerated {n_cities} random cities (seed={seed}):")
    for i, (x, y) in enumerate(cities):
        print(f"  City {i}: ({x:6.2f}, {y:6.2f})")
    
    # Create distance matrix
    matrix = create_distance_matrix(cities)
    print(f"\nDistance matrix created ({n_cities}x{n_cities})")
    
    # Run all algorithms
    print("\n" + "-" * 75)
    print("Running algorithms...")
    results = compare_algorithms(matrix)
    
    # Print comparison table
    print_comparison_table(results)
    print("-" * 75)
    
    # Find best solution
    best_name = min(results, key=lambda k: results[k][1])
    best_tour, best_dist, _ = results[best_name]
    
    print(f"\n✓ Best solution: {best_name}")
    print(f"  Tour: {best_tour}")
    print(f"  Distance: {best_dist:.2f}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_tour(cities, best_tour, best_dist)
    print("Saved to: tsp_tour.png")
    
    print("\n" + "=" * 75)
    print("Demo complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
