"""Function to load and parsing dataset files"""

def load_data(filename):

    labels = []
    features = []

    with open (filename, "r") as f:
        for line in f:
            parts = line.strip().split
            if not parts:
                continue
            labels.append(float(parts[0]))
            features.append([float(x) for x in parts[1:]])

    
    num_features = len(features[0])
    num_instances = len(labels)

    return labels, features, num_features, num_instances