import os
from src.preprocessing.load_data import load_dataset

if __name__ == "__main__":
    
    df = load_dataset() 
    print(df.shape)