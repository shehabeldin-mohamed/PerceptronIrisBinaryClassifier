import sys
import csv
from Perceptron import Perceptron

def load_dataset(filename):
    X = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = [float(x) for x in row[:-1]]
            X.append(features)
    return X

def load_labels(filename):
    y = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = 1 if row[-1] == "Iris-virginica" else 0
            y.append(label)
    return y

def manual_input(perceptron, num_features):
    try:
        user_input = input("\nEnter a feature vector separated by commas: ")
        vector = [float(x.strip()) for x in user_input.split(",")]

        if len(vector) != num_features:
            print(f"Error: Expected {num_features} features, but got {len(vector)}")
            return

        print(f"Vector Entered: {vector}")

        prediction = perceptron.predict(vector)
        print(f"Raw Prediction Output: {prediction}")

        species = "Iris-virginica" if prediction == 1 else "Iris-versicolor"
        print(f"Predicted Class: {species}")

    except ValueError:
        print("Invalid input! Please enter numeric values separated by commas.")


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <learning_rate> <train_file> <test_file>")
        return

    try:
        learning_rate = float(sys.argv[1])
        train_file = sys.argv[2]
        test_file = sys.argv[3]

        perceptron = Perceptron(learning_rate, epochs=1000)

        X_train = load_dataset(train_file)
        y_train = load_labels(train_file)
        X_test = load_dataset(test_file)
        y_test = load_labels(test_file)

        perceptron.train(X_train, y_train)
        accuracy = perceptron.accuracy(X_test, y_test)
        print(f"Test Set Accuracy: {accuracy:.2f}%")

        while True:
            manual_input(perceptron, len(X_train[0]))
            again = input("Test another? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting...")
                break

    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
