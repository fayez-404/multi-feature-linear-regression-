import pandas as pd
import numpy as np

def normalizing_features(df : pd.DataFrame):
    normalized_data = pd.DataFrame()
    for column in df.columns :
        if not pd.api.types.is_numeric_dtype(df[column]) :
            normalized_data[column] = df[column]
            continue
        std = df[column].std()
        normalized_data[column] = (df[column]-df[column].mean())/std
    return normalized_data

def normalizing_target(target : pd.Series):
    return (((target - target.mean() )/ target.std()))
    

def cost_function(weights,features:pd.DataFrame,target,b):
    weights = weights.reshape(len(features.columns),1)
    result = features.to_numpy() @ weights - target.to_numpy() + b
    return np.sum(result ** 2)/(2*len(target))

def gradient_descent(features:pd.DataFrame,target:pd.Series,iterations:int,learning_rate:float):
    weights = np.zeros(shape=(len(features.columns),1))
    b = 0 
    normalized_features = normalizing_features(features)
    normalized_target = normalizing_target(target)
    print(f"{"Iteration":>20}{"Cost":>20} {"step size":>15}",end=" ")
    for i in range(len(weights)) :
        print(f"{"w"+str(i):>10}",end=" ")
    print(f"{"b":>10} ")
    x = normalized_features.to_numpy()
    y = normalized_target.to_numpy().reshape(-1,1)
    n = len(normalized_target)
    for k in range(iterations):
        mal = (x @ weights ) - y + b
        cost = np.sum(mal ** 2) /(2*n)
        for i in range(len(normalized_features.columns)):
            stepsize = (np.sum(x[:,i:i+1] * mal)/n)*learning_rate
            weights[i][0] -= stepsize
        b -= (np.sum(mal)/n)*learning_rate
        if k % 100==0 :
            print(f"{k:20} {cost:20.3f} {stepsize:15.5f}",end=" ")
            for weight in weights :
                print(f"{weight[0]:10.3f}",end=" ")
            print(f"{b:10.3f} ")
    return (weights,b)
