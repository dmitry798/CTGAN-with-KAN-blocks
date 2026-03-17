"""
KAN_CTGAN Training Script for Credit Scoring Dataset
Обучение KAN_CTGAN модели на датасете кредитного скоринга
"""

import pandas as pd
import numpy as np
import pickle
from models.kan_ctgan_dp import KAN_CTGAN

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

# Определяем дискретные столбцы
discrete_columns = [
    'NumLoans',
    'NumRealEstateLoans', 
    'NumDependents',
    'Num30-59Delinquencies',
    'Num60-89Delinquencies',
    'Delinquent90'
]

print(f"\nDiscrete columns: {discrete_columns}")

# Инициализируем и обучаем KAN_CTGAN
print("\n" + "="*50)
print("KAN_CTGAN Training with Differential Privacy...")
print("="*50)

synthesizer = KAN_CTGAN(
    epochs=200,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=6e-4,
    discriminator_steps=2,
    # KAN specific parameters
    grid_size_gen=3,
    spline_order_gen=2,
    # Privacy parameters
    target_epsilon=5,  # Желаемый уровень приватности
    verbose=True,
    enable_gpu=True
)

print("Fitting KAN_CTGAN...")
synthesizer.fit(df_train, discrete_columns)

# Печатаем достигнутый уровень epsilon (если модель его отслеживает)
if hasattr(synthesizer, 'actual_epsilon'):
    print(f"\nAchieved epsilon: {synthesizer.actual_epsilon:.4f}")

# Генерируем синтетические данные
print("\nGenerating synthetic data...")
synthetic_data_kan_ctgan = synthesizer.sample(len(df_train))

# Сохраняем синтетические данные
output_path = '../synthetic_output_kan_ctgan_dp_credit_scoring.csv'
synthetic_data_kan_ctgan.to_csv(output_path, index=False)
print(f"Synthetic data saved to {output_path}")

# Сохраняем обученную модель
model_path = '../synthesizer_kan_ctgan_dp_credit_scoring.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(synthesizer, f)
    print(f"Model saved to {model_path}")

print("\n" + "="*50)
print("KAN_CTGAN Training Complete!")
print("="*50)
print(f"Original data shape: {df_train.shape}")
print(f"Synthetic data shape: {synthetic_data_kan_ctgan.shape}")
