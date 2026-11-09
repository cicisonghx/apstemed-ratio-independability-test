import pandas as pd

def load_data(excel_path):
    df = pd.read_excel(excel_path)
    return df
