"""
CTGAN Training Script for Credit Scoring Dataset
Обучение CTGAN модели на датасете кредитного скоринга
"""

import pickle
import pandas as pd
import numpy as np
from models.ctgan_dp import CTGAN

# Загружаем тренировочный датасет
print("Loading training data...")
df_train = pd.read_csv('../dataset/credit_scoring_train.csv')

print(f"Dataset shape: {df_train.shape}")
print(f"Columns: {df_train.columns.tolist()}")

# Удаляем client_id так как это просто индексный столбец
df_train = df_train.drop('client_id', axis=1)

# Обработка пропусков - заполняем медианой для continuous_columns
print("\nHandling missing values...")
for col in df_train.columns:
    if df_train[col].isna().any():
        if df_train[col].dtype in ['float64', 'int64']:
            df_train[col].fillna(df_train[col].median(), inplace=True)
        else:
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
        print(f"Filled missing values in {col}")

print(f"Final dataset shape: {df_train.shape}")
print(f"\nDataset info:")
print(df_train.info())

# Определяем дискретные столбцы - это целые числа, которые представляют категории
# или счетчики (NumLoans, NumRealEstateLoans, NumDependents, Num30-59Delinquencies, etc.)
discrete_columns = [
    'NumLoans',
    'NumRealEstateLoans', 
    'NumDependents',
    'Num30-59Delinquencies',
    'Num60-89Delinquencies',
    'Delinquent90'
]

print(f"\nDiscrete columns: {discrete_columns}")

# Инициализируем и обучаем CTGAN
print("\n" + "="*50)
print("CTGAN Training...")
print("="*50)

synthesizer = CTGAN(
    epochs=400,
    batch_size=512,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    verbose=True,
    enable_gpu=True,
    cuda=True
)

print("Fitting CTGAN...")
synthesizer.fit(df_train, discrete_columns)

# Генерируем синтетические данные
print("\nGenerating synthetic data...")
synthetic_data_ctgan = synthesizer.sample(len(df_train))

# Сохраняем синтетические данные
output_path = '../synthetic_output_ctgan_credit_scoring.csv'
synthetic_data_ctgan.to_csv(output_path, index=False)
print(f"Synthetic data saved to {output_path}")

# Сохраняем обученную модель
model_path = '../synthesizer_ctgan_credit_scoring.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(synthesizer, f)
    print(f"Model saved to {model_path}")

print("\n" + "="*50)
print("CTGAN Training Complete!")
print("="*50)
print(f"Original data shape: {df_train.shape}")
print(f"Synthetic data shape: {synthetic_data_ctgan.shape}")
