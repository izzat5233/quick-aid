import random

from qiskit import Aer, execute
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import VehicleRouting
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np


class QuantumWaterDeliveryVRP:
    def __init__(self, num_vehicles, depot, locations, demands):
        """
        Initialize Quantum VRP for water delivery

        Args:
            num_vehicles: Number of vehicles
            depot: Depot location [x, y]
            locations: List of delivery locations [[x1, y1], ...]
            demands: Water demand at each location
        """
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.locations = locations
        self.demands = demands
        self.distance_matrix = self._create_distance_matrix()

    def _create_distance_matrix(self):
        """Create distance matrix between all locations including depot"""
        all_points = [self.depot] + self.locations
        size = len(all_points)
        matrix = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                matrix[i][j] = np.sqrt(
                    (all_points[i][0] - all_points[j][0]) ** 2 +
                    (all_points[i][1] - all_points[j][1]) ** 2
                )

        return matrix.astype(int)

    def solve(self, p=1):
        """
        Solve VRP using QAOA

        Args:
            p: Number of QAOA layers
        """
        # Create VRP instance
        vrp = VehicleRouting(
            distance_matrix=self.distance_matrix,
            num_vehicles=self.num_vehicles
        )

        # Get quadratic program
        qp = vrp.to_quadratic_program()

        # Convert to QUBO
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(qp)

        # Set up QAOA
        optimizer = COBYLA()
        qaoa = QAOA(
            optimizer=optimizer,
            reps=p,
            quantum_instance=Aer.get_backend('qasm_simulator')
        )

        # Solve
        result = qaoa.compute_minimum_eigenvalue(qubo.to_ising()[0])

        # Convert result back to original problem
        raw_result = conv.interpret(result)

        # Extract routes from result
        routes = self._extract_routes(raw_result.x)

        return routes

    def _extract_routes(self, x):
        """Extract routes from QAOA result vector"""
        n = len(self.distance_matrix)
        routes = [[] for _ in range(self.num_vehicles)]

        # Convert binary solution to routes
        for i in range(n):
            for j in range(n):
                for v in range(self.num_vehicles):
                    idx = i * n * self.num_vehicles + j * self.num_vehicles + v
                    if idx < len(x) and x[idx] > 0.5:
                        routes[v].append((i, j))

        # Format routes
        formatted_routes = []
        total_distance = 0

        for v, route in enumerate(routes):
            if not route:
                continue

            # Sort route segments into proper order
            ordered_route = [0]  # Start at depot
            current = 0
            route_distance = 0

            while len(ordered_route) < len(route) + 1:
                for (i, j) in route:
                    if i == current and j not in ordered_route:
                        ordered_route.append(j)
                        route_distance += self.distance_matrix[current][j]
                        current = j
                        break

            formatted_routes.append({
                'vehicle': v,
                'route': ordered_route,
                'distance': route_distance
            })
            total_distance += route_distance

        return {
            'routes': formatted_routes,
            'total_distance': total_distance
        }


# Example usage with quantum approach
def generate_small_sample_data(num_locations=4):
    """Generate smaller sample data for quantum approach"""
    random.seed(42)
    depot = [0, 0]
    locations = [[random.randint(-20, 20), random.randint(-20, 20)]
                 for _ in range(num_locations)]
    demands = [random.randint(50, 200) for _ in range(num_locations + 1)]
    demands[0] = 0  # Depot has no demand

    return depot, locations, demands


# Run quantum example
depot, locations, demands = generate_small_sample_data()
quantum_vrp = QuantumWaterDeliveryVRP(
    num_vehicles=2,
    depot=depot,
    locations=locations,
    demands=demands
)

quantum_solution = quantum_vrp.solve()
print("\nQuantum VRP Solution:")
for route in quantum_solution['routes']:
    print(f"Vehicle {route['vehicle']}: {route['route']} (Distance: {route['distance']})")
print(f"Total distance: {quantum_solution['total_distance']}")
