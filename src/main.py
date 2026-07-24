from src.data.ingestion import load_data
from src.data.preprocessing import clean_data

df = load_data("data/raw/data.csv")
df = clean_data(df)

print(df.head())