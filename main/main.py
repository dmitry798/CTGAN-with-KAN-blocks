import pandas as pd
from my_ctgan_with_dp import KAN_CTGAN
# from my_ctgan import KAN_CTGAN


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
df = pd.read_csv(url, header=None, na_values='?')

# Дадим колонкам имена
df.columns = [f"A{i+1}" for i in range(df.shape[1])]

# Удалим строки с пропусками (CTGAN не любит NaN)
df = df.dropna().reset_index(drop=True)

# Непрерывные колонки по описанию датасета: A2, A3, A8, A11, A14, A15 [web:20]
continuous_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
for c in continuous_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=continuous_cols).reset_index(drop=True)

# Категориальные – по именам (соответствуют твоим индексам 0,3,4,5,6,8,9,11,12,15) [web:20]
discrete_columns = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13", "A16"]

synthesizer = KAN_CTGAN(
    epochs=200,
    verbose=True,
    grid_size_gen=5,
    spline_order_gen=3,
    grid_size_desc=5,
    spline_order_desc=3,
)

synthesizer.fit(df, discrete_columns)
synthetic_data = synthesizer.sample(1000)
synthetic_data.to_csv("synthetic_output.csv", index=False)


import matplotlib.pyplot as plt

real = df.copy()
synthetic = synthetic_data.copy()

col = "A2"  # числовой
plt.hist(real[col], bins=30, alpha=0.5, label="real", density=True)
plt.hist(synthetic[col], bins=30, alpha=0.5, label="synthetic", density=True)
plt.legend()
plt.show()


