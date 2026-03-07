from sklearn.datasets import fetch_openml
import numpy as np
import os

def load_data(dataset_name):
    cache_file = f"src/{dataset_name}_cache.npz"

    if os.path.exists(cache_file):
        print(f"Loading {dataset_name} from local cache...")
        data = np.load(cache_file)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    print(f"Downloading {dataset_name} from OpenML...") # Data is downloaded using openml in order to avoid using TensorFlow in the backend which is required by keras datasets.

    if dataset_name == 'mnist':
        dataset = fetch_openml('mnist_784', version=1, as_frame=False, parser="liac-arff")
    elif dataset_name == 'fashion_mnist':
        dataset = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser="liac-arff")
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.uint8)

    # Standard MNIST split: 60k train / 10k test
    X_train = X[:60000]
    y_train = y[:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    # Normalize
    X_train /= 255.0
    X_test /= 255.0

    # One-hot encode
    def one_hot(labels):
        out = np.zeros((labels.shape[0], 10))
        out[np.arange(labels.shape[0]), labels] = 1
        return out

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    return X_train, y_train, X_test, y_test