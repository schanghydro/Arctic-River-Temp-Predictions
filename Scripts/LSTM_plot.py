# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:26:27 2025

@author: sxc6234
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import scipy
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import geopandas as gpd
import contextily as cx
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch
import os
from scipy import stats
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

import os
os.chdir("./")


# Get Evaluation matrix/statistics for the time series where measure data >2 C
def read_results_2degree(p, epoch,threshold):
    run_dir = Path(p)
    with open(run_dir / "test" / epoch / "test_results.p", "rb") as fp:
        #results0a = pickle.load(fp)
        results0a=pd.read_pickle(fp)
    
    metrix0a=pd.read_csv(run_dir / "test" / epoch / "test_metrics.csv")
    #wshds=[list(resultsa.keys())][0]
    RMSE_l=[]
    PR_l=[]
    KGE_l=[]
    NSE_l=[]
    BIAS_l=[]
    for wshd in metrix0a.basin.to_list():
        qobs = results0a[str(wshd)]['1D']['xr']['mean_temp_c_obs'][:,-1]
        qsim = results0a[str(wshd)]['1D']['xr']['mean_temp_c_sim'][:,-1]

        
        qsim = qsim.where(qsim> threshold, drop=True)
        selected_coords = qsim.coords
        qobs = qobs.sel( date=selected_coords['date'])
        
        

        RMSE_l.append(metrics.rmse(qobs,qsim))
        PR_l.append(metrics.pearsonr(qobs,qsim))
        KGE_l.append(metrics.kge(qobs,qsim))
        NSE_l.append(metrics.nse(qobs,qsim))
        BIAS_l.append(np.nanmean(qsim.values - qobs.values))
        
    metrix0a["RMSE"]=RMSE_l
    metrix0a["Pearson-r"]=PR_l
    metrix0a["KGE"]=KGE_l
    metrix0a["NSE"]=NSE_l
    metrix0a["Bias"]=BIAS_l
    metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix_{}degree.csv".format(threshold))
    return(results0a, metrix0a )
          
# Get Evaluation matrix/statistics for the whole time-series, summer, winter, spring, and fall
def read_results(p, epoch,season):
    run_dir = Path(p)
    with open(run_dir / "test" / epoch / "test_results.p", "rb") as fp:
        #results0a = pickle.load(fp)
        results0a=pd.read_pickle(fp)
    
    metrix0a=pd.read_csv(run_dir / "test" / epoch / "test_metrics.csv")
    metrix0a=metrix0a.drop(columns="NSE")
    #wshds=[list(resultsa.keys())][0]
    RMSE_l=[]
    PR_l=[]
    KGE_l=[]
    #MISSEDP_l=[]
    NSE_l=[]
    num_l=[]
    BIAS_l=[]
    for wshd in metrix0a.basin.to_list():
        qobs = results0a[str(wshd)]['1D']['xr']['mean_temp_c_obs'][:,-1]
        qsim = results0a[str(wshd)]['1D']['xr']['mean_temp_c_sim'][:,-1]
        
        if season == "summer": 
            qobs = qobs.sel(date=(qobs['date.month'] >= 6) & (qobs['date.month'] <= 8))
            qsim = qsim.sel(date=(qsim['date.month'] >= 6) & (qsim['date.month'] <= 8))
        elif season == "spring": 
            qobs = qobs.sel(date=(qobs['date.month'] >=3) & (qobs['date.month'] <= 5))
            qsim = qsim.sel(date=(qsim['date.month'] >= 3) & (qsim['date.month'] <= 5))
        elif season == "fall": 
            qobs = qobs.sel(date=(qobs['date.month'] >=9) & (qobs['date.month'] <= 11))
            qsim = qsim.sel(date=(qsim['date.month'] >= 9) & (qsim['date.month'] <=11))
        elif season == "winter": 
            qobs = qobs.sel(date=(qobs['date.month'] >=12) | (qobs['date.month'] <= 2))
            qsim = qsim.sel(date=(qsim['date.month'] >= 12) | (qsim['date.month'] <=2))
        else:
            a="all"
        if qobs.sum().values == 0:
            RMSE_l.append(np.nan)
            PR_l.append(np.nan)
            KGE_l.append(np.nan)
            NSE_l.append(np.nan)
            num_l.append(np.nan)
            BIAS_l.append(np.nan)

        else:

            dic=metrics.calculate_all_metrics(qobs,qsim)
            RMSE_l.append(dic["RMSE"])
            PR_l.append(dic["Pearson-r"])
            KGE_l.append(dic["KGE"])
            NSE_l.append(dic["NSE"])
            qobs_d=qobs_d=qobs.dropna( dim="date", how="all")
            num_l.append(qobs_d.shape[0])
            BIAS_l.append(np.nanmean(qsim.values - qobs.values))
    metrix0a["RMSE"]=RMSE_l
    metrix0a["Pearson-r"]=PR_l
    metrix0a["KGE"]=KGE_l
    metrix0a["NSE"]=NSE_l
    metrix0a["Num_obs"]=num_l
    metrix0a["Bias"]=BIAS_l
    
    if season =="summer":
        metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix_summer.csv")
    elif season =="winter":
        metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix_winter.csv")
    elif season =="spring":
        metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix_spring.csv")
    elif season =="fall":
        metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix_fall.csv")
    elif season =="all":
        metrix0a.to_csv(run_dir / "test" / epoch /"all_evaluation_metrix.csv")
    return(results0a, metrix0a )        
    
read_results("./runs/Final_20230921_2909_103637/", "model_epoch008","summer")
read_results("./runs/Final_20230921_2909_103637/", "model_epoch008","spring")
read_results("./runs/Final_20230921_2909_103637/", "model_epoch008","winter")
read_results("./runs/Final_20230921_2909_103637/", "model_epoch008","fall")
results0a, metrix0a =read_results("./runs/Final_20230921_2909_103637/", "model_epoch008","all")
read_results_2degree("./runs/Final_20230921_2909_103637/", "model_epoch008",2)


#%%
# The five evaluation matrix of stream temperature predictions for the test basins.
df=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix.csv",index_col=0)
df_summer=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_summer.csv",index_col=0)
df_spring=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_spring.csv",index_col=0)
df_fall=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_fall.csv",index_col=0)
df_winter=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_winter.csv",index_col=0)
df_2c=pd.read_csv("./runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_2degree.csv",index_col=0)

#mask out the two gauges which have less than 90 records along the 22 yeas 
df=df[~df.basin.isin([111,121]) ]
df_summer=df_summer[~df_summer.basin.isin([111,121]) ]
df_spring=df_spring[~df_spring.basin.isin([111,121]) ]
df_fall=df_fall[~df_fall.basin.isin([111,121]) ]
df_winter=df_winter[~df_winter.index.isin([111,121]) ]
df_2c=df_2c[~df_2c.basin.isin([111,121]) ]
df["season"]=1
df_2c["season"]=2
df_spring["season"]=3
df_summer["season"]=4
df_fall["season"]=5
df_winter["season"]=6

frames = [df, df_2c, df_spring, df_summer, df_fall]

result = pd.concat(frames)


def boxplot(x,y,color,data,y1,y2):
    fig, ax = plt.subplots(figsize=(5,7))
    # Define a color palette
    palette = {
        1: '#0c4767',  # red
        2: '#b9a44c',  # green
        3: '#566e3d',    # blue
        4: '#6f2dbd', 
        5: '#fe9920', 
        }
    
    box1=sns.boxplot( data=result, x=x, y=y, linewidth=2,notch=True,medianprops={"color": "black"},width=0.4,  palette=palette, showfliers = False)
    for patch in box1.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 1))
    #ax.axhline(y=0, color='gray', linestyle='--',linewidth=2.5)
    ax.set_ylim([y1,y2])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.xticks(range(0,5),["All",">2C", "Spr","Sum","Fall"],fontsize=22)
    #plt.xticks([1],["Test"],fontsize=20)
    plt.yticks( fontsize=22)
    #ax.axhline(y=0, color='r', linestyle='--')
    box1.set_ylabel(y, fontsize=24)
    #box1.set_xlabel('Test', fontsize=24)
x="season"
y="RMSE"
data=result
y1=0
y2=2.8
color="#6d6875"
boxplot(x,y,color,data,y1,y2)

x="season"
y="NSE"
data=result
y1=-0.2
y2=1
color="#00b4d8"
boxplot(x,y,color,data,y1,y2)

x="season"
y="Pearson-r"
data=result
y1=0.6
y2=1
color="#606c38"
boxplot(x,y,color,data,y1,y2)

x="season"
y="KGE"
data=result
y1=0.4
y2=1
color="#e29578"
boxplot(x,y,color,data,y1,y2)

x="season"
y="Bias"
data=result
y1=-2.5
y2=2.5
color="#8d99ae"
boxplot(x,y,color,data,y1,y2)




#%%
#plot Time-series (in-situ temperatrue V.S. simulated temperatures) for each test catchment

run_dir = Path('./runs/Final_20230921_2909_103637/')
with open(run_dir / "test" / "model_epoch008" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

df=df.sort_values(by=["NSE"], ascending=True)
for i in list(df.index):
    wshd=str(i)
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(8,4))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
    # extract observations and simulations
    qobs = results[wshd]['1D']['xr']['mean_temp_c_obs'][:,-1]
    qsim = results[wshd]['1D']['xr']['mean_temp_c_sim'][:,-1]
    nse= round(df_2c.loc[i,"NSE"],3)
    
    ax.plot(qobs['date'], qobs,markerfacecolor="#0077b6",markersize= 3, linestyle='none', marker='o',markeredgewidth=0)
    ax.plot(qsim['date'], qsim,'k-',linewidth=0.5)
    #ax.set_ylabel("Temperature (C)",fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=13)
    ax.set_title("Watershed = {}, NSE = {}".format(wshd, nse),fontsize=16)
    #ax.set_title(f"Test period - NSE {results['{}']['1D']['NSE']:.3f}".fromat(wshd))

    fig.tight_layout() 



#%%
#Analyze the model performance of changing state 

wshds=map(int,['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8'])
overl=[]
underl=[]

for wshd in wshds:
    
    
    qobs = results[str(wshd)]['1D']['xr']['mean_temp_c_obs'][:,-1]
    qsim = results[str(wshd)]['1D']['xr']['mean_temp_c_sim'][:,-1]
    
    qobs = qobs.sel(date=(qobs['date.month'] >=11) | (qobs['date.month'] <= 3)).to_dataframe()
    qsim = qsim.sel(date=(qsim['date.month'] >= 11) | (qsim['date.month'] <=3)).to_dataframe()  
    qobs['mean_temp_c_sim']=qsim['mean_temp_c_sim'].values
    total=qobs['mean_temp_c_sim'].notna().sum()
    # Condition 1: Observed ≤ 0 while Simulated > 0
    condition1 = qobs[(qobs["mean_temp_c_obs"] <= 0) & (qobs["mean_temp_c_sim"] > 0)]
    over=condition1.shape[0]
    # Condition 2: Simulated ≤ 0 while Observed > 0
    condition2 = qobs[(qobs["mean_temp_c_sim"] <= 0) & (qobs["mean_temp_c_obs"] > 0)]
    under=condition2.shape[0]
    # Display results
    print("Condition 1: Observed ≤ 0 while Simulated > 0")
    print(over/total*100);
    
    print("\nCondition 2: Simulated ≤ 0 while Observed > 0")
    print(under/total*100)
    
    overl.append(over/total*100)
    underl.append(under/total*100)
    
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(3,4))
plt.boxplot([overl, underl],showfliers=False)

# Add title and labels
#plt.title("Box Plot of Two Lists")
#plt.ylabel("Values")

stats=pd.DataFrame({"over":overl, "under":underl})