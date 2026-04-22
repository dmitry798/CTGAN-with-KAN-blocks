import pickle
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, na_values='?')

df.columns = ['age','workclass','fnlwgt','education','education-num',
              'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','income']

df = df.dropna().reset_index(drop=True) # CTGAN doesn't work with NaN

discrete_columns = ['workclass','education','marital-status','occupation',
                    'relationship','race','sex','native-country','income']

with open('synthesizer_kan_ctgan_dp.pkl', 'rb') as f:
    synthesizer = pickle.load(f)

synthetic_data = synthesizer.sample(1000)

col = "hours-per-week"
plt.hist(df[col], bins=30, alpha=0.5, label="real", density=True)
plt.hist(synthetic_data[col], bins=30, alpha=0.5, label="synthetic", density=True)
plt.legend()
plt.show()
