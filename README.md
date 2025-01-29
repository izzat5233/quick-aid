## Problem: Water Needs

### Restrictions

- Each group needs certain amount of water in a limited time window (their water will run out after that time).
- Routes are of different lengths (different time to travel).

### Goal

The goal is to find an optimal route for the vehicle, this route could be special in a certain way, for example:
- Total Travel Time (Fastest Delivery). This is not the most important factor. Dismiss.
- Total Distance Traveled (Fuel Efficiency & Coverage). Fuel is not important. Dismiss.
- Least Unserved Demand (Fulfillment Efficiency). This is the most important factor. This is the goal.
   - Minimizes the number of people left without water within their critical time window.
   - Take into account the water supplies available for each group and the time before they run out of water.
   - Take into account how many vehicles are available and how much water they carry (each one can take a different route).