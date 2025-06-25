import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('gesture_recognition.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

df['label'] = df['label'].map({'punch': 0, 'flex': 1})

df.fillna(df.mean(), inplace=True)

scaler = StandardScaler()
df[['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']] = scaler.fit_transform(df[['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']])

def augment_data(df):
    augmented_df = pd.DataFrame()
    for index, row in df.iterrows():
        noisy_row = row.copy()
        noisy_row['aX'] += np.random.normal(0, 0.1)
        noisy_row['aY'] += np.random.normal(0, 0.1)
        noisy_row['aZ'] += np.random.normal(0, 0.1)
        noisy_row['gX'] += np.random.normal(0, 1)
        noisy_row['gY'] += np.random.normal(0, 1)
        noisy_row['gZ'] += np.random.normal(0, 1)
        augmented_df = augmented_df.append(noisy_row, ignore_index=True)
    return augmented_df

augmented_df = augment_data(df)
df = pd.concat([df, augmented_df], ignore_index=True)

X = df.drop(['label'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
