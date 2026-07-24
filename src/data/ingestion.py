import pandas as pd


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except: 
        print("yo, chech the file path --> {file_path}")
    