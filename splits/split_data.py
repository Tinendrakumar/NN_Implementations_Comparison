import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the dataset
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]
df = pd.read_csv('../agaricus-lepiota.data', header=None, names=column_names)

# Mapping class labels to binary values
df['class'] = df['class'].map({'p': 0, 'e': 1})

# Creating one-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=column_names[1:])

# Reorder columns to have class_0 and class_1 at the beginning
df_encoded = pd.concat([df_encoded[['class']], df_encoded.drop(['class'], axis=1)], axis=1)

# Convert boolean columns to 0 or 1
df_encoded = df_encoded.astype(int)

# Split the dataset into training, validation, and testing sets
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Save the datasets to separate files
train_df.to_csv('training.txt', index=False, header=False)
val_df.to_csv('validation.txt', index=False, header=False)
test_df.to_csv('testing.txt', index=False, header=False)

# Save the entire dataset to a CSV file
df_encoded.to_csv('train(visual).csv', index=False)
