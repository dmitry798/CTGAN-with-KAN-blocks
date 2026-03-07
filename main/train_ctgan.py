import pickle
import pandas as pd
from ctgan import CTGAN

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

synthesizer = CTGAN(
    epochs=500,
    verbose=True
)

synthesizer.fit(df, discrete_columns)
synthetic_data_ctgan = synthesizer.sample(len(df))
synthetic_data_ctgan.to_csv("synthetic_output_ctgan.csv", index=False)


with open('synthesizer_ctgan.pkl', 'wb') as f:
    pickle.dump(synthesizer, f)
    print("Модель ctgan успешно сохранена!")

