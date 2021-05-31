#%% Import
import pandas as pd
import numpy as np

#%% Despike Function
# a is the ratio  of how similar in magnitude the rise and fall of a spike have to be to be considered a spike
# n is the maximum duration of the spike 
# a in (0,1], 1 is most conservative.
# n is positive whole number, 1 is most conservative.
def despike(df,n,a):
    for col in df.columns:
        for i in df.index:
            for k in range(1,n+1):
                if k+i in df.index and df[col][i] != 0:
                    if -df[col][i+k]/df[col][i]>=a and -df[col][i+k]/df[col][i]<=1/a and all(df[col][j] == 0 for j in range(i+1,i+k)):
                        df.at[i,col]=0
                        df.at[i+k,col]=0
                        #print("altering at "+str(col)+" "+str(i))
                        
                    
                

#%% Load files and tidy up

df = pd.read_json("dancing_2dJoints_nosmooth.json")
df = df.transpose()
df = df.applymap(lambda x: np.array(x))
dfdiff = df.diff(axis=0, periods = 1)
dfdiff=dfdiff.drop(1)
dfdiff0 = dfdiff.applymap(lambda x: x[0])
dfdiff1 = dfdiff.applymap(lambda x: x[1])
dfdiff2 = dfdiff.applymap(lambda x: x[2])


#%%
despike(dfdiff0,2,0.9)
despike(dfdiff1,2,0.9)
despike(dfdiff2,1,0.9)


#%% Remerge and Cumsum
dfl=[dfdiff0,dfdiff1,dfdiff2]
for col in dfdiff0.columns:
    for i in dfdiff0.index:
        dfdiff.at[i,col]=np.array([dfr[col][i] for dfr in dfl])
dfdiff = dfdiff.append(df.loc[1])
dfdiff = dfdiff.sort_index()       
dfs = dfdiff.cumsum()

#%% Back to original format and save
dfs = dfs.applymap(lambda x: x.tolist())
dfs = dfs.transpose()
dfs.to_json("dancing_2dJoints_smooth.json")
