import os
import joblib


def save_object(obj, file_path):
    print(f"file_path = {file_path}")
    print(f"directory = {os.path.dirname(file_path)}")

    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Load any saved object.
    """

    return joblib.load(file_path)

