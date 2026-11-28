import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from multi_feature_linear_regression import normalizing_features , normalizing_target, gradient_decent
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "example data" / "Data.csv"

raw_data = pd.read_csv(DATA_PATH).loc[:,["Country","League","Player Names","Matches_Played","Goals","xG","Shots","OnTarget","Year"]].astype({"Year":str})

normalized_data = normalizing_features(raw_data)

# features are onTarget xG shots and goals is the target 
#init values
 
features = ["xG","Shots","OnTarget"]
target = "Goals"
learning_rate =0.01

(weights,b)=gradient_decent(raw_data.loc[:,features],raw_data[target],300,learning_rate)

fig,ax = plt.subplots(figsize=(12,4))
ax.plot(raw_data.loc[:50,"Player Names"],raw_data.loc[:50,target])
predicted_values = normalized_data.loc[:,features].to_numpy()@weights +b
ax.plot(normalized_data.loc[:50,"Player Names"],(predicted_values.reshape(1,-1)[0][:51]*raw_data[target].std())+raw_data[target].mean())
plt.xticks(rotation="vertical")
plt.show()

