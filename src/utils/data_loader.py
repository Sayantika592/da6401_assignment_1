import numpy as np
import os
import gzip
import urllib.request


def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28*28).astype(np.float32) / 255.0


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def one_hot(labels):
    out = np.zeros((labels.shape[0], 10))
    out[np.arange(labels.shape[0]), labels] = 1
    return out


def load_data(dataset_name):

    cache_file = f"src/{dataset_name}_cache.npz"

    if os.path.exists(cache_file):
        print(f"Loading {dataset_name} from local cache...")
        data = np.load(cache_file)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    if dataset_name == "fashion_mnist":

        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz"
        }

    elif dataset_name == "mnist":

        base_url = "http://yann.lecun.com/exdb/mnist/"

        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz"
        }

    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    os.makedirs("data", exist_ok=True)

    for file in files.values():
        download(base_url + file, f"data/{file}")

    X_train = load_mnist_images(f"data/{files['train_images']}")
    y_train = load_mnist_labels(f"data/{files['train_labels']}")

    X_test = load_mnist_images(f"data/{files['test_images']}")
    y_test = load_mnist_labels(f"data/{files['test_labels']}")

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