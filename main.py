"""Main .py for the Feature Selection Algorithm."""

import time

from data_loader import load_data
from search import forward_selection, backward_elimination


def main():
    print("Welcome to Feature Selection Algorithm.\n")

    filename = input("Type in the name of the file to test: ")
    labels, features, num_features, num_instances = load_data(filename)

    print(f"\nThis dataset has {num_features} features "
          f"(not including the class attribute), with {num_instances} instances.")

    print("\nType the number of the algorithm you want to run.\n")
    print("    1) Forward Selection")
    print("    2) Backward Elimination\n")
    choice = input("").strip()

    start_time = time.time()

    if choice == '1':
        forward_selection(labels, features, num_features)
    elif choice == '2':
        backward_elimination(labels, features, num_features)
    else:
        print("Invalid choice.")
        return

    elapsed = time.time() - start_time
    print(f"\nTime elapsed: {elapsed:.1f} seconds")


if __name__ == '__main__':
    main()