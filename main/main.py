import pandas as pd
# from my_ctgan_with_dp import KAN_CTGAN
# from my_ctgan import KAN_CTGAN
from fix_my_ctgan import KAN_CTGAN

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, na_values='?')

# Дадим колонкам имена
df.columns = ['age','workclass','fnlwgt','education','education-num',
              'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','income']

# Удалим строки с пропусками (CTGAN не любит NaN)
df = df.dropna().reset_index(drop=True)

# # Непрерывные колонки по описанию датасета: A2, A3, A8, A11, A14, A15 [web:20]
# continuous_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
# for c in continuous_cols:
#     df[c] = pd.to_numeric(df[c], errors="coerce")

# df = df.dropna(subset=continuous_cols).reset_index(drop=True)

# Категориальные – по именам (соответствуют твоим индексам 0,3,4,5,6,8,9,11,12,15) [web:20]
discrete_columns = ['workclass','education','marital-status','occupation',
                    'relationship','race','sex','native-country','income']

# synthesizer = KAN_CTGAN(
#     epochs=500,
#     verbose=True,
#     grid_size_gen=5,
#     spline_order_gen=2
# )

# synthesizer.fit(df, discrete_columns)
# synthetic_data = synthesizer.sample(1000)
# synthetic_data.to_csv("synthetic_output.csv", index=False)



import matplotlib.pyplot as plt

real = df.copy()
# synthetic = synthetic_data.copy()

import pickle
with open('synthesizer.pkl', 'rb') as f:
    synthesizer = pickle.load(f)

synthetic_data = synthesizer.sample(1000)

col = "hours-per-week"
plt.hist(df[col], bins=30, alpha=0.5, label="real", density=True)
plt.hist(synthetic_data[col], bins=30, alpha=0.5, label="synthetic", density=True)
plt.legend()
plt.show()


