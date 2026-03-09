# ==============================
# Disease Diagnosis AI Project
# BFS + Machine Learning Models
# ==============================

# -------- Graph Definition --------
graph = {
    'Fever': ['Flu', 'Malaria', 'Dengue', 'Food Poisoning', 'Measles'],
    'Food Poisoning': ['Fever'],
    'Flu': ['Cough', 'Fatigue', 'Fever', 'Chills'],
    'Cough': ['Measles', 'Flu'],
    'Fatigue': ['Flu', 'Dengue'],
    'Chills': ['Flu', 'Malaria'],
    'Measles': ['Cough', 'Rash', 'Fever'],
    'Rash': ['Measles', 'Dengue'],
    'Malaria': ['Headache', 'Fever', 'Chills'],
    'Headache': ['Dengue', 'Malaria'],
    'Dengue': ['Fatigue', 'Rash', 'Fever', 'Headache']
}


# -------- BFS Algorithm --------
def BFS(graph, start, goal):

    queue = [[start]]
    visited = []

    while queue:

        path = queue.pop(0)
        node = path[-1]

        if node in visited:
            continue

        visited.append(node)

        if node == goal:
            return path

        for neighbor in graph.get(node, []):
            queue.append(path + [neighbor])

    return None


# -------- Import Libraries --------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------- Load Dataset --------
def load_data():

    data = pd.read_csv("Disease_Data.csv")

    X = data.drop(columns="Outcome")
    y = data["Outcome"]

    return train_test_split(X, y, test_size=0.15, random_state=0)


# -------- Decision Tree Model --------
def decision_tree_model(X_train, X_test, y_train, y_test):

    model = DecisionTreeClassifier(max_depth=2, random_state=0)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions) * 100,
        "Precision": precision_score(y_test, predictions, average='weighted') * 100,
        "Recall": recall_score(y_test, predictions, average='weighted') * 100,
        "F1": f1_score(y_test, predictions, average='weighted') * 100
    }

    return results


# -------- Backpropagation Model --------
def mlp_model(X_train, X_test, y_train, y_test):

    model = MLPClassifier(
        hidden_layer_sizes=(10,10),
        max_iter=1000,
        random_state=0,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        alpha=0.02,
        early_stopping=True
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions) * 100,
        "Precision": precision_score(y_test, predictions, average='weighted') * 100,
        "Recall": recall_score(y_test, predictions, average='weighted') * 100,
        "F1": f1_score(y_test, predictions, average='weighted') * 100
    }

    return results


# -------- Main Program --------
def main():

    X_train, X_test, y_train, y_test = load_data()

    dt_results = decision_tree_model(X_train, X_test, y_train, y_test)
    mlp_results = mlp_model(X_train, X_test, y_train, y_test)

    start_symptom = "Dengue"
    end_symptom = "Cough"

    path = BFS(graph, start_symptom, end_symptom)


    print("1. Decision Tree Results")
    for k,v in dt_results.items():
        print(f"   {k}: {v:.2f}%")


    print("\n2. Backpropagation (MLP) Results")
    for k,v in mlp_results.items():
        print(f"   {k}: {v:.2f}%")


    print("\n3. BFS Path Search")

    if path:
        print(f"   Path from {start_symptom} to {end_symptom}: {path}")
    else:
        print("   No path found")


# -------- Run Program --------
if __name__ == "__main__":
    main()