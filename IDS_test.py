# Import Libraries
import pickle
import sys
import pandas as pd

# Read Dataset
df = pd.read_csv(sys.argv[1])
X = df[df.columns[0]]

# Columns selected in feature extraction and selection based on Z-Score
df = df[['L_T1',  'L_T6', 'L_T7', 'F_PU2', 'S_PU2', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2', 'P_J280', 'P_J415', 'P_J302', 'P_J306', 'P_J307', 'P_J14']]

# Load Model from pickle file
dbfile = open('model.pickle', 'rb')     
model = pickle.load(dbfile)

# Making Prection
result = model.predict(df)

# Typecasting int labels to str labels
result = pd.Series(result).replace({0:"NORMAL",1:"ATTACK"})

# Recombining the CSV
frame = { X.name : X, 'RESULT': result}
result_df = pd.DataFrame(frame)

# Exporting CSV
result_df.to_csv("result.csv", index= False)