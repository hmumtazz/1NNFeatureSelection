"""Feature Selection Search Algorithms"""

from classifier import leave_eval, default_rater


def _format_feature_set(feature_set):
    return '{' + ','.join(str(x) for x in sorted(feature_set)) + '}'


def forward_selection(labels, features, num_features):
    """Greedy forward selection search."""

    all_features = set(range(1, num_features + 1))
    all_accuracy = leave_eval(labels, features, all_features)

    print(f"\nRunning nearest neighbor with all {num_features} features, "
          f'using "leaving-one-out" evaluation, I get an accuracy of '
          f"{all_accuracy * 100:.1f}%\n")

    dr = default_rater(labels)
    print("Beginning search.\n")
    print(f"Using no features (default rate), accuracy is {dr * 100:.1f}%\n")

    current_set = set()
    best_overall_set = set()
    best_overall_accuracy = 0.0
    prev_best_accuracy = dr

    for level in range(num_features):
        best_feature = None
        best_accuracy = 0.0

        remaining = [f for f in range(1, num_features + 1) if f not in current_set]

        for f in remaining:
            candidate = current_set | {f}
            accuracy = leave_eval(labels, features, candidate)
            print(f"Using feature(s) {_format_feature_set(candidate)} "
                  f"accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = f

        current_set.add(best_feature)

        if best_accuracy < prev_best_accuracy:
            print("\n(Accuracy has decreased! Continuing search in case of local maxima)")

        print(f"Feature set {_format_feature_set(current_set)} was best, "
              f"accuracy is {best_accuracy * 100:.1f}%\n")

        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = set(current_set)

        prev_best_accuracy = best_accuracy

    print(f"Finished search!! The best feature subset is {_format_feature_set(best_overall_set)}, "
          f"which has an accuracy of {best_overall_accuracy * 100:.1f}%")

    return best_overall_set, best_overall_accuracy


def backward_elimination(labels, features, num_features):
    """Greedy backward elimination search."""
    current_set = set(range(1, num_features + 1))
    all_accuracy = leave_eval(labels, features, current_set)

    print(f"\nRunning nearest neighbor with all {num_features} features, "
          f'using "leaving-one-out" eval, I get an accurcy of '
          f"{all_accuracy * 100:.1f}%\n")

    best_overall_accuracy = all_accuracy
    best_overall_set = set(current_set)
    prev_best_accuracy = all_accuracy

    print("start search.\n")

    for level in range(num_features - 1):
        best_feature_to_remove = None
        best_accuracy = 0.0

        for f in sorted(current_set):
            candidate = current_set - {f}
            accuracy = leave_eval(labels, features, candidate)
            print(f"Using feature(s) {_format_feature_set(candidate)} "
                  f"accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_to_remove = f

        current_set.remove(best_feature_to_remove)

        if best_accuracy < prev_best_accuracy:
            print("\n(Accuracy has decreased! Continuing search in case of local maxima)")

        print(f"Feature set {_format_feature_set(current_set)} was best, "
              f"accuracy is {best_accuracy * 100:.1f}%\n")

        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_set = set(current_set)

        prev_best_accuracy = best_accuracy

    dr = default_rater(labels)
    if dr < prev_best_accuracy:
        print("(Accuracy has decreased! Continuing search in case of local maxima)")
    print(f"Using no features (default rate), accuracy is {dr * 100:.1f}%\n")

    print(f"Finished search!! The best feature subset is {_format_feature_set(best_overall_set)}, "
          f"which has an accuracy of {best_overall_accuracy * 100:.1f}%")

    return best_overall_set, best_overall_accuracy