import pandas as pd
import numpy as np
from ctgan import CTGAN
import pickle
import os

# Load data:
base_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
df = pd.read_csv(os.path.join(base_dir, 'credit_scoring_train.csv'))

print(f"{df.shape[0]} rows × {df.shape[1]} columns")
print(f"target 'Delinquent90': {df['Delinquent90'].value_counts().to_dict()}")

# Data preprocessing:
print(f"number of NaN before preprocessing: {df.isna().sum().sum()}")

if 'client_id' in df.columns:
    df = df.drop('client_id', axis=1)

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Column'{col}' : NaN filled with median {median_val:.2f}")

print(f"number of NaN after preprocessing: {df.isna().sum().sum()}")
print(f"shape: {df.shape}")

discrete_columns = []


synthesizer = CTGAN(
    # Architecture params:
    embedding_dim=128,
    generator_dim=(512, 512),
    discriminator_dim=(256, 256),
    
    # Training params:
    generator_lr=2e-4,
    generator_decay=1e-3,
    discriminator_lr=6e-4,
    discriminator_decay=1e-5,
    batch_size=250, # batch_size % pac = 0
    discriminator_steps=2,
    
    # General params:
    log_frequency=True,
    verbose=True,
    pac=10,
    cuda=True
)

# Training:
print(f"\nTraining started:")
synthesizer.fit(df, discrete_columns)
print(f"\nTraining finished!")

# Synthetic data generation:
synthetic_data_ctgan = synthesizer.sample(len(df))

# Save:
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'synthetic_output_ctgan_credit_scoring.csv')
synthetic_data_ctgan.to_csv(output_path, index=False)
print(f"\nSynthetic data saved: {output_path}")

weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, 'synthesizer_ctgan_credit_scoring_optimized.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(synthesizer, f)
print(f"Model saved: {model_path}")
