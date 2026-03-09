import pandas as pd
from models.kan_ctgan_dp import KAN_CTGAN
import pickle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, na_values='?')

# Дадим колонкам имена
df.columns = ['age','workclass','fnlwgt','education','education-num',
              'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','income']

# Удалим строки с пропусками (CTGAN не любит NaN)
df = df.dropna().reset_index(drop=True)

# Категориальные – по именам (соответствуют твоим индексам 0,3,4,5,6,8,9,11,12,15) [web:20]
discrete_columns = ['workclass','education','marital-status','occupation',
                    'relationship','race','sex','native-country','income']

synthesizer = KAN_CTGAN(
    epochs=200,
    verbose=True,
    grid_size_gen=3,
    spline_order_gen=2
)

synthesizer.fit(df, discrete_columns)
print(f"Достигнутый epsilon: {synthesizer.actual_epsilon:.4f}")
synthetic_data_kan_ctgan_dp = synthesizer.sample(len(df))
synthetic_data_kan_ctgan_dp.to_csv("synthetic_output_kan_ctgan_dp.csv", index=False)

with open('synthesizer_kan_ctgan_dp.pkl', 'wb') as f:
    pickle.dump(synthesizer, f)
    print("Модель kan_ctgan_dp успешно сохранена!")
