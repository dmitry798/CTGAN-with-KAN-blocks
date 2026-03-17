"""
KAN_CTGAN Testing Script for Credit Scoring Dataset
Тестирование KAN_CTGAN модели на датасете кредитного скоринга
"""

import pickle
import pandas as pd
import numpy as np
from models.kan_ctgan_dp import KAN_CTGAN

# Загружаем тестовый датасет
print("Loading test data...")
df_test = pd.read_csv('../dataset/credit_scoring_test.csv')

print(f"Test dataset shape: {df_test.shape}")
print(f"Columns: {df_test.columns.tolist()}")

# Удаляем client_id так как это просто индексный столбец
df_test = df_test.drop('client_id', axis=1)

# Обработка пропусков
print("\nHandling missing values...")
for col in df_test.columns:
    if df_test[col].isna().any():
        if df_test[col].dtype in ['float64', 'int64']:
            df_test[col].fillna(df_test[col].median(), inplace=True)
        else:
            df_test[col].fillna(df_test[col].mode()[0], inplace=True)
        print(f"Filled missing values in {col}")

print(f"Final test dataset shape: {df_test.shape}")

# Загружаем обученную модель
print("\n" + "="*50)
print("Loading trained KAN_CTGAN model...")
print("="*50)

model_path = '../synthesizer_kan_ctgan_dp_credit_scoring.pkl'
try:
    with open(model_path, 'rb') as f:
        synthesizer = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please run train_kan_ctgan_dp_credit_scoring.py first to train the model.")
    exit(1)

# Печатаем информацию о модели, если доступна
if hasattr(synthesizer, 'actual_epsilon'):
    print(f"Achieved epsilon (privacy level): {synthesizer.actual_epsilon:.4f}")

# Генерируем синтетические данные на основе обученной модели
print("\nGenerating synthetic data using trained model...")
num_samples = len(df_test)
synthetic_test_data = synthesizer.sample(num_samples)

print(f"Generated synthetic data shape: {synthetic_test_data.shape}")

# Сохраняем результаты
output_path = '../synthetic_test_output_kan_ctgan_dp_credit_scoring.csv'
synthetic_test_data.to_csv(output_path, index=False)
print(f"Synthetic test data saved to {output_path}")

# Выводим статистику сравнения
print("\n" + "="*50)
print("Comparison Statistics")
print("="*50)
print("\nOriginal Test Data Description:")
print(df_test.describe())

print("\nSynthetic Test Data Description:")
print(synthetic_test_data.describe())

# Сравниваем корреляции
print("\n" + "="*50)
print("Correlation Analysis")
print("="*50)

numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()

print("\nOriginal Data Correlations (first 5x5):")
corr_original = df_test[numeric_cols].corr()
print(corr_original.iloc[:5, :5])

print("\nSynthetic Data Correlations (first 5x5):")
corr_synthetic = synthetic_test_data[numeric_cols].corr()
print(corr_synthetic.iloc[:5, :5])

print("\n" + "="*50)
print("KAN_CTGAN Testing Complete!")
print("="*50)
