import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Generate Synthetic Data
def generate_data(num_samples=5000, num_groups=3, num_trucks=5):
    np.random.seed(42)
    data = []

    for _ in range(num_samples):
        # Generate groups' needs
        groups = []
        for _ in range(num_groups):
            group = {
                'water_need': np.random.randint(0, 100),
                'water_time': np.random.randint(1, 24),
                'food_need': np.random.randint(0, 100),
                'food_time': np.random.randint(1, 24),
                'med_need': np.random.randint(0, 100),
                'med_time': np.random.randint(1, 24),
                'distance': np.random.uniform(1, 50)
            }
            groups.append(group)

        # Generate trucks
        trucks = []
        for _ in range(num_trucks):
            truck = {
                'water': np.random.randint(0, 100),
                'food': np.random.randint(0, 100),
                'medicine': np.random.randint(0, 100)
            }
            trucks.append(truck)

        # Calculate priority scores for each truck
        scores = []
        for truck in trucks:
            score = 0

            # Calculate group weights
            for group in groups:
                # Water priority calculation
                water_urgency = (group['water_need'] / 100) * (1 / group['water_time']) * (1 / group['distance'])
                food_urgency = (group['food_need'] / 100) * (1 / group['food_time']) * (1 / group['distance'])
                med_urgency = (group['med_need'] / 100) * (1 / group['med_time']) * (1 / group['distance'])

                # Apply emergency override for time < 2 hours
                if group['water_time'] < 2:
                    water_urgency *= 10
                if group['food_time'] < 2:
                    food_urgency *= 10
                if group['med_time'] < 2:
                    med_urgency *= 10

                score += (truck['water'] * water_urgency +
                          truck['food'] * food_urgency +
                          truck['medicine'] * med_urgency)

            scores.append(score)

        # Find best truck
        best_truck = np.argmax(scores)

        # Create features (aggregate group needs)
        water_emergency = sum(1 / g['water_time'] for g in groups if g['water_time'] < 2)
        food_emergency = sum(1 / g['food_time'] for g in groups if g['food_time'] < 2)
        med_emergency = sum(1 / g['med_time'] for g in groups if g['med_time'] < 2)

        for i, truck in enumerate(trucks):
            features = {
                'truck_water': truck['water'],
                'truck_food': truck['food'],
                'truck_med': truck['medicine'],
                'water_emergency': water_emergency,
                'food_emergency': food_emergency,
                'med_emergency': med_emergency,
                'is_best': 1 if i == best_truck else 0
            }
            data.append(features)

    return pd.DataFrame(data)


# Generate dataset
df = generate_data()

# 2. Train-Test Split
X = df.drop('is_best', axis=1)
y = df['is_best']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
history = model.fit(X_train, y_train)

# 4. Evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.2f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.2f}")

# 5. Plot Feature Importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()


# 6. Test Case Analysis with Plots and Explanations (3 Trucks)
def plot_test_case(test_truck, situation, probabilities):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(test_truck)), probabilities, color=['blue', 'orange', 'green'])

    # Add truck information to the plot
    for i, bar in enumerate(bars):
        truck_info = f"Water: {test_truck[i]['truck_water']}\nFood: {test_truck[i]['truck_food']}\nMedicine: {test_truck[i]['truck_med']}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, truck_info,
                 ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.xticks(range(len(test_truck)), [f"Truck {i + 1}" for i in range(len(test_truck))])
    plt.ylabel("Selection Probability")
    plt.title(f"Situation: {situation}")
    plt.ylim(0, 1.1)
    plt.show()


def analyze_test_case(trucks, situation):
    print(f"\nSituation: {situation}")
    print("Trucks and their supplies:")
    for i, truck in enumerate(trucks):
        print(f"Truck {i + 1}: Water={truck['truck_water']}, Food={truck['truck_food']}, Medicine={truck['truck_med']}")

    test_truck_df = pd.DataFrame(trucks)
    probabilities = model.predict_proba(test_truck_df)[:, 1]

    print("\nSelection Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Truck {i + 1}: {prob:.2f}")

    plot_test_case(trucks, situation, probabilities)

    # Explanation
    best_truck = np.argmax(probabilities) + 1
    print(f"\nExplanation: Truck {best_truck} is selected because it best matches the situation.")
    if "water emergency" in situation.lower():
        print("Water emergencies are prioritized, so trucks with more water are favored.")
    if "food emergency" in situation.lower():
        print("Food emergencies are prioritized, so trucks with more food are favored.")
    if "medicine emergency" in situation.lower():
        print("Medicine emergencies are prioritized, so trucks with more medicine are favored.")
    print("-" * 50)


# Define test cases with 3 trucks
test_cases = [
    {
        "situation": "High water emergency (water_emergency=5)",
        "trucks": [
            {"truck_water": 80, "truck_food": 50, "truck_med": 30, "water_emergency": 5, "food_emergency": 0,
             "med_emergency": 0},
            {"truck_water": 30, "truck_food": 80, "truck_med": 50, "water_emergency": 5, "food_emergency": 0,
             "med_emergency": 0},
            {"truck_water": 60, "truck_food": 60, "truck_med": 60, "water_emergency": 5, "food_emergency": 0,
             "med_emergency": 0}
        ]
    },
    {
        "situation": "High food emergency (food_emergency=5)",
        "trucks": [
            {"truck_water": 80, "truck_food": 50, "truck_med": 30, "water_emergency": 0, "food_emergency": 5,
             "med_emergency": 0},
            {"truck_water": 30, "truck_food": 80, "truck_med": 50, "water_emergency": 0, "food_emergency": 5,
             "med_emergency": 0},
            {"truck_water": 40, "truck_food": 90, "truck_med": 40, "water_emergency": 0, "food_emergency": 5,
             "med_emergency": 0}
        ]
    },
    {
        "situation": "High medicine emergency (med_emergency=5)",
        "trucks": [
            {"truck_water": 80, "truck_food": 50, "truck_med": 30, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 5},
            {"truck_water": 30, "truck_food": 80, "truck_med": 50, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 5},
            {"truck_water": 50, "truck_food": 50, "truck_med": 90, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 5}
        ]
    },
    {
        "situation": "Mixed emergency (water_emergency=3, food_emergency=2)",
        "trucks": [
            {"truck_water": 80, "truck_food": 50, "truck_med": 30, "water_emergency": 3, "food_emergency": 2,
             "med_emergency": 0},
            {"truck_water": 30, "truck_food": 80, "truck_med": 50, "water_emergency": 3, "food_emergency": 2,
             "med_emergency": 0},
            {"truck_water": 70, "truck_food": 70, "truck_med": 40, "water_emergency": 3, "food_emergency": 2,
             "med_emergency": 0}
        ]
    },
    {
        "situation": "No emergency, balanced supplies",
        "trucks": [
            {"truck_water": 50, "truck_food": 50, "truck_med": 50, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 0},
            {"truck_water": 60, "truck_food": 40, "truck_med": 40, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 0},
            {"truck_water": 40, "truck_food": 60, "truck_med": 60, "water_emergency": 0, "food_emergency": 0,
             "med_emergency": 0}
        ]
    }
]

# Run test cases
for case in test_cases:
    analyze_test_case(case["trucks"], case["situation"])
