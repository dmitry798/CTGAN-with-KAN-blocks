import pickle
import pandas as pd
import matplotlib.pyplot as plt

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

with open('synthesizer_ctgan.pkl', 'rb') as f:
    synthesizer = pickle.load(f)

synthetic_data = synthesizer.sample(1000)

col = "hours-per-week"
plt.hist(df[col], bins=30, alpha=0.5, label="real", density=True)
plt.hist(synthetic_data[col], bins=30, alpha=0.5, label="synthetic", density=True)
plt.legend()
plt.show()
