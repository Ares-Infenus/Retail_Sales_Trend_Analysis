import pandas as pd

df = pd.read_csv(
    r"D:\Portafolio oficial\Retail Sales Trend Analysis\data\data\processed\clear_train.csv"
)
a = df.info()
print(a)
