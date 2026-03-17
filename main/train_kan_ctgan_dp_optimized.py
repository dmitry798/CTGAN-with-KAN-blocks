"""
Оптимизированное обучение KAN-CTGAN-DP на Credit Scoring датасете
с лучшими гиперпараметрами для класса данных с дисбалансом
"""

import pandas as pd
import numpy as np
from models.kan_ctgan_dp import KAN_CTGAN
import pickle
import os

# =====================================================================
# ЗАГРУЗКА ДАННЫХ
# =====================================================================
base_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
df = pd.read_csv(os.path.join(base_dir, 'credit_scoring_train.csv'))

print(f"✓ Загружено данных: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"✓ Целевая переменная 'Delinquent90': {df['Delinquent90'].value_counts().to_dict()}")

# =====================================================================
# ПРЕДОБРАБОТКА ДАННЫХ (обработка NaN)
# =====================================================================
print(f"\n📊 Предобработка данных:")
print(f"   NaN значений ДО: {df.isna().sum().sum()}")

# Удалим client_id (как в анализе)
if 'client_id' in df.columns:
    df = df.drop('client_id', axis=1)

# Заполняем NaN медианой для каждого столбца
# (CTGAN не поддерживает null values)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   ✓ Столбец '{col}': заполнены NaN медианой {median_val:.2f}")

print(f"   NaN значений ПОСЛЕ: {df.isna().sum().sum()}")
print(f"   Форма данных: {df.shape}")

# Категориальные колонки (если есть)
discrete_columns = []

# =====================================================================
# ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ДЛЯ НЕСБАЛАНСИРОВАННЫХ ДАННЫХ
# =====================================================================
print("\n🔧 Инициализация оптимизированного KAN-CTGAN-DP с параметрами:")
print("   • target_epsilon=5 (больше утечки приватности → выше качество)")
print("   • grid_size_gen=5 (гибче аппроксимирует распределения)")
print("   • generator_dim=(512, 512) (мощнее генератор)")
print("   • batch_size=256 (лучше сходимость на малом классе)")
print("   • discriminator_steps=2 (крепче дискриминатор)")
print("   • epochs=400 (больше обучения)")

synthesizer = KAN_CTGAN(
    # ===== KAN параметры (главное!) =====
    grid_size_gen=5,           # Увеличиваем для лучшей аппроксимации
    spline_order_gen=3,        # Оставляем, работает хорошо
    target_epsilon=5,         # ГЛАВНОЕ: больше epsilon = лучше качество (но меньше приватности)
    
    # ===== Архитектура =====
    embedding_dim=128,
    generator_dim=(512, 512),  # Более мощный генератор
    discriminator_dim=(256, 256),
    
    # ===== Обучение =====
    generator_lr=2e-4,
    generator_decay=1e-3,
    discriminator_lr=6e-4,
    discriminator_decay=1e-5,
    batch_size=256,            # Меньше для лучшей конвергенции
    discriminator_steps=2,     # Сильнее дискриминатор
    
    # ===== Общие =====
    log_frequency=True,
    verbose=True,
    epochs=400,                # Больше эпох для лучшей сходимости
    pac=1,                     # Обязательно 1 для opacus
    enable_gpu=True
)

# =====================================================================
# ОБУЧЕНИЕ
# =====================================================================
print("\n📚 Начало обучения (может занять 10-20 минут)...")
synthesizer.fit(df, discrete_columns)

# =====================================================================
# РЕЗУЛЬТАТЫ И СОХРАНЕНИЕ
# =====================================================================
print(f"\n✅ Обучение завершено!")
print(f"✓ Достигнутый epsilon: {synthesizer.actual_epsilon:.4f}")
print(f"✓ Target epsilon было: 12")

# Генерируем синтетические данные
synthetic_data_kan_ctgan_dp = synthesizer.sample(len(df))

# Сохраняем
output_path = os.path.join(base_dir, '..', 'synthetic_output_kan_ctgan_dp_credit_scoring_optimized.csv')
synthetic_data_kan_ctgan_dp.to_csv(output_path, index=False)
print(f"\n💾 Синтетические данные сохранены: {output_path}")

model_path = os.path.join(base_dir, '..', 'synthesizer_kan_ctgan_dp_credit_scoring_optimized.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(synthesizer, f)
print(f"💾 Модель сохранена: {model_path}")

print("\n✨ Готово! Теперь запусти анализ в new_utility.ipynb с новыми данными")
