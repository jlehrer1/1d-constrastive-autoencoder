import pandas as pd
# Load the data
df = pd.read_csv('local_inventory.csv')
# Print the first few rows of the DataFrame
print(df.head())
# Print the data types of the columns
print(df.dtypes)
# Print the shape of the DataFrame
print(df.shape)
#Name the columns
df.columns = ['Time','path']
#Convert the path column to a string
df['path'] = df['path'].astype(str)
print(df.shape)
print(df.head())
#Use Regular Expressions to Extract Information from Strings
#Only conserve the rows that have kilosort2/derived in the path
df = df[df['path'].str.contains('derived/kilosort2')]
# Print the shape of the DataFrame
print(df.head())
print(df.shape)
#Only conserve the rows that have the word 'phy_zip
df = df[df['path'].str.contains('phy.zip')]
print(df.shape)
df.drop(['Time'], axis=1, inplace=True)
#Write the path column to a txt file
df.to_csv('phy_zip.txt', header=None, index=None, sep=' ', mode='a')