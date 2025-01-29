import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random


class WaterDeliveryVRP:
    def __init__(self, num_vehicles, depot, locations, demands, time_windows):
        """
        Initialize the Water Delivery VRP problem

        Args:
            num_vehicles: Number of available vehicles
            depot: Location of the depot [x, y]
            locations: List of delivery locations [[x1, y1], [x2, y2], ...]
            demands: Water demand at each location in liters
            time_windows: List of [earliest, latest] delivery times for each location
        """
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.locations = locations
        self.demands = demands
        self.time_windows = time_windows

        # Convert locations to distance matrix
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

    def solve(self, vehicle_capacity=1000):
        """
        Solve the VRP problem

        Args:
            vehicle_capacity: Maximum water capacity per vehicle in liters
        """
        manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix),
            self.num_vehicles,
            0  # depot
        )
        routing = pywrapcp.RoutingModel(manager)

        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node] if from_node > 0 else 0

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [vehicle_capacity] * self.num_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )

        # Add time window constraints
        def time_callback(from_index, to_index):
            # Convert distance to time (assume 1 unit distance = 1 minute)
            return distance_callback(from_index, to_index)

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30,  # allow waiting time
            300,  # maximum time per vehicle
            False,  # don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # Add time windows for each location
        for location_idx, time_window in enumerate(self.time_windows):
            if location_idx == 0:
                continue  # Skip depot
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(
                time_window[0],
                time_window[1]
            )

        # Set first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            return self._get_solution(manager, routing, solution)
        return None

    def _get_solution(self, manager, routing, solution):
        """Extract and format the solution"""
        routes = []
        total_distance = 0

        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            route.append(manager.IndexToNode(index))
            routes.append({
                'vehicle': vehicle_id,
                'route': route,
                'distance': route_distance
            })
            total_distance += route_distance

        return {
            'routes': routes,
            'total_distance': total_distance
        }


# Example usage
def generate_sample_data(num_locations=10):
    """Generate sample data for testing"""
    random.seed(42)
    depot = [0, 0]
    locations = [[random.randint(-50, 50), random.randint(-50, 50)]
                 for _ in range(num_locations)]
    demands = [random.randint(50, 200) for _ in range(num_locations + 1)]
    demands[0] = 0  # Depot has no demand
    time_windows = [[random.randint(0, 150), random.randint(151, 300)]
                    for _ in range(num_locations + 1)]
    time_windows[0] = [0, 300]  # Depot is always open

    return depot, locations, demands, time_windows


# Run the example
depot, locations, demands, time_windows = generate_sample_data()
vrp = WaterDeliveryVRP(
    num_vehicles=3,
    depot=depot,
    locations=locations,
    demands=demands,
    time_windows=time_windows
)

solution = vrp.solve(vehicle_capacity=1000)
print("\nClassical VRP Solution:")
for route in solution['routes']:
    print(f"Vehicle {route['vehicle']}: {route['route']} (Distance: {route['distance']})")
print(f"Total distance: {solution['total_distance']}")