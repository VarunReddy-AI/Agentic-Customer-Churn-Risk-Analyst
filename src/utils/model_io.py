import os
import joblib


def save_object(obj, file_path):
    """
    Save any Python object.
    """

    directory = os.path.dirname(file_path)

    os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, file_path)

    print(f"Saved: {file_path}")


def load_object(file_path):
    """
    Load any saved object.
    """

    return joblib.load(file_path)