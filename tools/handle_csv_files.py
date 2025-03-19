import pandas as pd


df = pd.read_csv('../csvs/london_merged.csv')

# Convert 'timestamp' column to datetime and extract features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
# df['day'] = df['timestamp'].dt.day
# df['hour'] = df['timestamp'].dt.hour

# Drop the original 'timestamp' column
df.drop('timestamp', axis=1, inplace=True)
df.to_csv('../csvs/Clean_london_merged.csv')

# numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# numerical_data = df[numerical_columns].fillna(df.median(numeric_only=True))

# Save the cleaned data to a new CSV file
# numerical_data.to_csv("Cleaned_data_houses.csv", index=False)
