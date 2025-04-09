# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:45:43 2023

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

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
#%%
#Train the LSTM model using the configuration file
if __name__ == '__main__':
        # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        print("yes gpu")
        torch.device('cuda:1')
        start_run(config_file=Path("./Data/Hyperparameter_{}.yml".format(18)))



#%%
#Evaluate the model on both train and test sets

run_dir = Path('./runs/Final_20230921_2909_103637/')
eval_run(run_dir=run_dir, period="train")
eval_run(run_dir=run_dir, period="test")

#%%
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

df=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix.csv",index_col=0)
df_summer=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_summer.csv",index_col=0)
df_spring=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_spring.csv",index_col=0)
df_fall=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_fall.csv",index_col=0)
df_winter=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_winter.csv",index_col=0)
df_2c=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/test/model_epoch008/all_evaluation_metrix_2degree.csv",index_col=0)

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

with open(run_dir / "test" / "model_epoch008" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)
    
results.keys()


#%%


wshds=map(int,['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8'])
    
p="G:/Shared drives/InteRFACE River Temperature (No CUI)/Data/AKTemp_all_atts.csv"    
attdf=pd.read_csv(p,index_col=0)
attdf=attdf[attdf.index.isin(wshds) ]  

df=df.set_index("basin")
df_spring=df_spring.set_index("basin")
df_summer=df_summer.set_index("basin")
df_fall=df_fall.set_index("basin")
df_winter=df_winter.set_index("basin")
df_2c=df_2c.set_index("basin")

 
attdf = attdf.reindex(list(df.index.values))

attdf["2c_nse"]=df_2c["NSE"]
attdf["2c_rmse"]=df_2c["RMSE"]
#attdf["2C_num"]=df_2c["Num_obs"]
attdf["all_nse"]=df["NSE"]
attdf["all_rmse"]=df["RMSE"]
attdf["summer_nse"]=df_summer["NSE"]
attdf["fall_nse"]=df_fall["NSE"]
attdf["winter_nse"]=df_winter["NSE"]
attdf["spring_nse"]=df_spring["NSE"]
attdf["summer_num"]=df_summer["Num_obs"]
attdf["fall_num"]=df_fall["Num_obs"]
attdf["winter_num"]=df_winter["Num_obs"]
attdf["spring_num"]=df_spring["Num_obs"]
attdf_s1=attdf[["all_nse","2c_nse","summer_nse","spring_nse","fall_nse", "winter_nse","summer_num","spring_num","fall_num", "winter_num", "latitude", "longitude", "series_count_days","series_start_datetime", "series_end_datetime"]]



gdf = gpd.GeoDataFrame( attdf, geometry=gpd.points_from_xy(attdf_s1.longitude, attdf_s1.latitude))
gdf=gdf.set_crs('EPSG:4236')
gdf=gdf.to_crs("EPSG:3338")
#gdf.to_file(r"D:\ShuyuChang\AKTemp\Data\Final_results\final_results.shp")
ak_bd=r"D:\ShuyuChang\AKTemp\Data\AK_state_boundary\AK_State_BD.shp"
ak_bd=gpd.read_file(ak_bd)
ak_bd=ak_bd.to_crs("EPSG:3338")

#%%
'''
fig, ax = plt.subplots(figsize=(10, 10))#

ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf.plot(ax=ax, marker='o',column='2c_nse',cmap="plasma",  markersize=100,edgecolor ="black",vmin=0.4, vmax=1)
ax.set_axis_off()
fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\figrue4a_nse_2c.png", format="png", dpi=300)
'''

gdf_t=gdf.sort_values(by=['all_nse'], ascending=False)
fig, ax = plt.subplots(figsize=(15, 15))#
ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf_t2.plot(ax=ax, marker='o',column='2c_nse',cmap="plasma",  markersize=15,edgecolor ="black",vmin=0.5, vmax=1, legend=True,legend_kwds={'orientation': "horizontal"})
gdf_t1.plot(ax=ax, marker="^", markersize=40,color="k")
ax.set_axis_off()
fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\f2.png", format="png", dpi=1100)


fig, ax = plt.subplots(figsize=(10, 10))#
gdf_t=gdf.sort_values(by=['2c_nse'], ascending=False)
gdf_t1=gdf_t[gdf_t['2c_nse']<=0.5]
gdf_t2=gdf_t[gdf_t['2c_nse']>0.5]
ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')

gdf_t2.plot(ax=ax, marker='o',column='2c_nse',cmap="plasma",  markersize=40,edgecolor ="black",vmin=0.5, vmax=1, legend=True,legend_kwds={'orientation': "horizontal"})
gdf_t1.plot(ax=ax, marker="^", markersize=40,color="k")
ax.set_axis_off()
fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\f1.png", format="png", dpi=300)
#%%
gdf_sorted_rmse=gdf_t.sort_values(by="2c_rmse")
gdf_t1_rmse=gdf_sorted_rmse.loc[[244,240,308,272],:]
gdf_t2_rmse=gdf_sorted_rmse[~gdf_sorted_rmse.index.isin([244,240,308,272])]
fig, ax = plt.subplots(figsize=(10, 10))#
gdf_t=gdf.sort_values(by=['2c_rmse'], ascending=False)

ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf_sorted_rmse.plot(ax=ax, marker='o',column='2c_rmse',cmap="Blues",  markersize=80,edgecolor ="black",vmin=0.4, vmax=2, legend=True,legend_kwds={'orientation': "horizontal"})
#gdf_t2_rmse.plot(ax=ax, marker='o',column='2c_rmse',cmap="Blues",  markersize=80,edgecolor ="black",vmin=0.4, vmax=2, legend=True,legend_kwds={'orientation': "horizontal"})
#gdf_t1_rmse.plot(ax=ax, marker="^", markersize=50,color="k")
ax.set_axis_off()
#fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\f1.png", format="png", dpi=300)


#%%
fig, ax = plt.subplots(figsize=(10, 10))#

ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf.plot(ax=ax, marker='o',column='summer_nse',cmap="plasma",  markersize=60,edgecolor ="black",vmin=0.5, vmax=1)
ax.set_axis_off()
fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\figrue4b_nse.png", format="png", dpi=300)

gdf_t=gdf.sort_values(by=['summer_nse'], ascending=False)
fig, ax = plt.subplots(figsize=(15, 15))#
ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf_t.plot(ax=ax, marker='o',column='summer_nse',cmap="plasma",  markersize=20,edgecolor ="black",vmin=0.5, vmax=1)
ax.set_axis_off()
#fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\figrue4b_nse_1.png", format="png", dpi=600)


fig, ax = plt.subplots(figsize=(10, 10))#

ak_bd.plot(ax=ax, edgecolor='k', linewidth =1, facecolor='none')
gdf.plot(ax=ax, marker='o',column='summer_nse',cmap="plasma",  markersize=10,edgecolor ="black",vmin=0.5, vmax=1, legend=True,legend_kwds={'orientation': "horizontal"})
ax.set_axis_off()
#fig.savefig(r"D:\ShuyuChang\AKTemp\Figrues\figrue4b_nse_legend.png", format="png", dpi=300)


#%%
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
    fig.savefig("D:/ShuyuChang/AKTemp/Figrues/time_series_plot/"+wshd+".png", format="png", dpi=300)



#%%

import datetime as dt
i=244
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
plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
fig.tight_layout() 
fig.savefig("D:/ShuyuChang/AKTemp/Figrues/time_series_plot/"+wshd+".png", format="png", dpi=300)

#%%
#linear regression using absolute numbers 


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
y = attdf['2c_nse'].values

attl=["topo_GM90_topo_idx", "ha_ppd_pk_uav", "ha_for_pc_use","ha_snd_pc_uav", "ha_snd_pc_uav",\
      "ha_snd_pc_uav", "ice_PastickAK_probability", "norm_aet","ice_GLIMS_glacier", "ha_cmi_ix_uyr",\
          "norm_dis", "ha_ele_mt_uav", "ha_snw_pc_uyr", "latitude", "ha_tmp_dc_syr","ha_cly_pc_uav",\
              "area_km2","ha_wet_pc_ug1", "ha_nli_ix_uav", "norm_riverarea", "ha_pnv_cl_smj",\
                  "ha_lit_cl_smj", "soil_SG250_soc_15_30", "soil_Pelletier_sed_thickness",\
                      "ha_lka_pc_sse", "ha_tmp_dc_s06", "ha_tmp_dc_s07", "ha_tmp_dc_s08","ha_tmp_dc_s09"]

r2l=[]
coefl=[]
namesl=[]
pl=[]
# For each x variable, perform linear regression and get R^2
for x_var in attdf.columns[:-1]:  # Exclude the 'y' column
    if x_var in attl:
        X = attdf[x_var].values.reshape(-1, 1) 
        # Convert to 2D array
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.reshape(42,),y)
        print(x_var +": {}".format(r2))
        print(model.coef_)
        print(p_value)
        print("")
        
        r2l.append(r2)
        coefl.append(model.coef_[0])
        namesl.append(x_var)
        pl.append(p_value)
sorted_indices = sorted(range(len(r2l)), key=lambda k: r2l[k], reverse=True)
sorted_r2l = [r2l[i] for i in sorted_indices]
sorted_coefl = [coefl[i] for i in sorted_indices]
sorted_names = [namesl[i] for i in sorted_indices]
sorted_p= [pl[i] for i in sorted_indices]
plt.plot(np.log(attdf['norm_dis'].values),np.log(attdf['2c_rmse'].values), "*")

text_list = [""]*27
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(7,4))
ax.bar(range(1,28), sorted_r2l, width=0.8,  align='center',edgecolor="k",color="#61a5c2")
ax.set_xlabel("Basin Attributes", fontsize=13)
ax.set_ylabel("R2", fontsize=13)
ax.set_xticks(range(1,28),text_list )

fig.savefig("D:/ShuyuChang/AKTemp/Figrues/r2_2cnse_att.png", format="png", dpi=300)


#short_names=['Norm Q','River Area %', 'Sub Lake Area %',' Lithology','Norm AET']
short_names=['River Area %', 'Elevation',' Snow %','Norm Q', "Lithology"]
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(6,4))
ax.barh( range(1,6),sorted_r2l[0:5],  height=0.5,align='center',edgecolor="k",color="#3d348b")
ax.set_xlabel("Basin Attributes", fontsize=13)
ax.set_ylabel("R2", fontsize=13)
ax.set_yticks(range(1,6),short_names ,fontsize=15)

plt.gca().invert_yaxis()
fig.savefig("D:/ShuyuChang/AKTemp/Figrues/r2_2cnse_att_l.png", format="png", dpi=300)

for i in range (5):
    att=sorted_names[i]
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(3,3))
    ax.plot(attdf[att], y, 'o', markeredgecolor ='k', markerfacecolor='#9a8c98',markersize=12,alpha=0.8)
    fig.savefig("D:/ShuyuChang/AKTemp/Figrues/att_{}.png".format(i), format="png", dpi=300)


#%%
#linear regression using log log  numbers 


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
y = attdf['2c_nse'].values -attdf['2c_nse'].values.min()+0.01
y=np.log10(y)
attl=["topo_GM90_topo_idx", "ha_ppd_pk_uav", "ha_for_pc_use","ha_snd_pc_uav", "ha_snd_pc_uav",\
      "ha_snd_pc_uav", "ice_PastickAK_probability", "norm_aet","ice_GLIMS_glacier", "ha_cmi_ix_uyr",\
          "norm_dis", "ha_ele_mt_uav", "ha_snw_pc_uyr", "latitude", "ha_tmp_dc_syr","ha_cly_pc_uav",\
              "area_km2","ha_wet_pc_ug1", "ha_nli_ix_uav", "norm_riverarea", "ha_pnv_cl_smj",\
                  "ha_lit_cl_smj", "soil_SG250_soc_15_30", "soil_Pelletier_sed_thickness",\
                      "ha_lka_pc_sse", "ha_tmp_dc_s06", "ha_tmp_dc_s07", "ha_tmp_dc_s08","ha_tmp_dc_s09"]

r2l=[]
coefl=[]
namesl=[]
# For each x variable, perform linear regression and get R^2
for x_var in attdf.columns[:-1]:  # Exclude the 'y' column
    if x_var in attl:
        X = attdf[x_var].values.reshape(-1, 1) 
        X=X-X.min()+0.01
        X=np.log10(X)
        # Convert to 2D array
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        print(x_var +": {}".format(r2))
        print(model.coef_)
        print("")
        
        r2l.append(r2)
        coefl.append(model.coef_[0])
        namesl.append(x_var)
sorted_indices = sorted(range(len(r2l)), key=lambda k: r2l[k], reverse=True)
sorted_r2l = [r2l[i] for i in sorted_indices]

sorted_names = [namesl[i] for i in sorted_indices]
plt.plot(np.log(attdf['norm_dis'].values),np.log(attdf['2c_rmse'].values), "*")

text_list = [""]*27
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(7,4))
ax.bar(range(1,28), sorted_r2l, width=0.8,  align='center',edgecolor="k",color="#61a5c2")
ax.set_xlabel("Basin Attributes", fontsize=13)
ax.set_ylabel("R2", fontsize=13)
ax.set_xticks(range(1,28),text_list )

fig.savefig("D:/ShuyuChang/AKTemp/Figrues/r2_2cnse_att.png", format="png", dpi=300)


#short_names=['Norm Q','River Area %', 'Sub Lake Area %',' Lithology','Norm AET']
short_names=['River Area %', 'Elevation',' Snow %','Norm Q', "Lithology"]
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(6,4))
ax.barh( range(1,6),sorted_r2l[0:5],  height=0.5,align='center',edgecolor="k",color="#3d348b")
ax.set_xlabel("Basin Attributes", fontsize=13)
ax.set_ylabel("R2", fontsize=13)
ax.set_yticks(range(1,6),short_names ,fontsize=15)

plt.gca().invert_yaxis()
fig.savefig("D:/ShuyuChang/AKTemp/Figrues/r2_2cnse_att_l.png", format="png", dpi=300)

for i in range (5):
    att=sorted_names[i]
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(3,3))
    ax.loglog(attdf[att], y, 'o', markeredgecolor ='k', markerfacecolor='#9a8c98',markersize=12,alpha=0.8)
    fig.savefig("D:/ShuyuChang/AKTemp/Figrues/att_{}.png".format(i), format="png", dpi=300)




#%%
att=pd.read_csv("G:\Shared drives\InteRFACE River Temperature (No CUI)\Data\AKTemp_all_atts.csv")



attl=["topo_GM90_topo_idx", "ha_ppd_pk_uav", "ha_for_pc_use","ha_snd_pc_uav", "ha_cly_pc_uav",\
      "ha_slt_pc_uav", "ice_PastickAK_probability", "norm_aet","ice_GLIMS_glacier", "ha_cmi_ix_uyr",\
          "norm_dis", "ha_ele_mt_uav", "ha_snw_pc_uyr", "latitude", "ha_tmp_dc_syr","ha_cly_pc_uav",\
              "area_km2","ha_wet_pc_ug1", "ha_nli_ix_uav", "norm_c", "ha_pnv_cl_smj",\
                  "ha_lit_cl_smj", "soil_SG250_soc_15_30", "soil_Pelletier_sed_thickness",\
                      "ha_lka_pc_sse", "ha_tmp_dc_s06", "ha_tmp_dc_s07", "ha_tmp_dc_s08","ha_tmp_dc_s09"]




att=att[attl]
summar=att.describe()


#%%
############################################################################
###Analyze the model performance of changing state 
########################################################################

epoch="model_epoch008"
run_dir = Path("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637")
with open(run_dir / "test" / epoch / "test_results.p", "rb") as fp:
    #results0a = pickle.load(fp)
    results0a=pd.read_pickle(fp)


wshds=map(int,['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8'])
overl=[]
underl=[]

for wshd in wshds:
    
    
    qobs = results0a[str(wshd)]['1D']['xr']['mean_temp_c_obs'][:,-1]
    qsim = results0a[str(wshd)]['1D']['xr']['mean_temp_c_sim'][:,-1]
    
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
#%%    

def read_results_2degree(p, epoch,threshold):
    run_dir = Path(p)
    with open(run_dir / "train" / epoch / "train_results.p", "rb") as fp:
        #results0a = pickle.load(fp)
        results0a=pd.read_pickle(fp)
    
    metrix0a=pd.read_csv(run_dir / "train" / epoch / "train_metrics.csv")
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
    metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix_{}degree.csv".format(threshold))
    return(results0a, metrix0a )
          

def read_results(p, epoch,season):
    run_dir = Path(p)
    with open(run_dir / "train" / epoch / "train_results.p", "rb") as fp:
        #results0a = pickle.load(fp)
        results0a=pd.read_pickle(fp)
    
    metrix0a=pd.read_csv(run_dir / "train" / epoch / "train_metrics.csv")
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
        if qobs.count()<5:
            RMSE_l.append(np.nan)
            PR_l.append(np.nan)
            KGE_l.append(np.nan)
            NSE_l.append(np.nan)
            num_l.append(np.nan)
            BIAS_l.append(np.nan)

        else:
        #mp=metrics.missed_peaks(qobs,qsim)
            dic=metrics.calculate_all_metrics(qobs,qsim)
            RMSE_l.append(dic["RMSE"])
            PR_l.append(dic["Pearson-r"])
            KGE_l.append(dic["KGE"])
            NSE_l.append(dic["NSE"])
            qobs_d=qobs_d=qobs.dropna( dim="date", how="all")
            num_l.append(qobs_d.shape[0])
            BIAS_l.append(np.nanmean(qsim.values - qobs.values))
      

        #MISSEDP_l.append(mp)

    metrix0a["RMSE"]=RMSE_l
    metrix0a["Pearson-r"]=PR_l
    metrix0a["KGE"]=KGE_l
    #metrix0a["Missed Peaks"]=MISSEDP_l
    metrix0a["NSE"]=NSE_l
    metrix0a["Num_obs"]=num_l
    metrix0a["Bias"]=BIAS_l
    if season =="summer":
        metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix_summer.csv")
    elif season =="winter":
        metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix_winter.csv")
    elif season =="spring":
        metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix_spring.csv")
    elif season =="fall":
        metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix_fall.csv")
    elif season =="all":
        metrix0a.to_csv(run_dir / "train" / epoch /"all_evaluation_metrix.csv")
    return(results0a, metrix0a )        
    
read_results("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008","summer")
read_results("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008","spring")
read_results("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008","winter")
read_results("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008","fall")
results0a, metrix0a =read_results("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008","all")
read_results_2degree("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637", "model_epoch008",2)

#%%

df_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix.csv",index_col=0)
df_summer_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix_summer.csv",index_col=0)
df_spring_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix_spring.csv",index_col=0)
df_fall_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix_fall.csv",index_col=0)
df_winter_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix_winter.csv",index_col=0)
df_2c_train=pd.read_csv("D:/ShuyuChang/AKTemp/Output/Final_20230921/runs/Final_20230921_2909_103637/train/model_epoch008/all_evaluation_metrix_2degree.csv",index_col=0)


df_train["season"]=1
df_2c_train["season"]=2
df_spring_train["season"]=3
df_summer_train["season"]=4
df_fall_train["season"]=5
df_winter_train["season"]=6

frames = [df_train, df_2c_train, df_spring_train, df_summer_train, df_fall_train]

result = pd.concat(frames)

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




from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
y = attdf['2c_rmse'].values 
y=np.log10(y)
attl=["topo_GM90_topo_idx", "ha_ppd_pk_uav", "ha_for_pc_use","ha_snd_pc_uav","ha_cly_pc_uav",\
      "ha_slt_pc_uav", "ice_PastickAK_probability", "norm_aet","ice_GLIMS_glacier", "ha_cmi_ix_uyr",\
          "norm_dis", "ha_ele_mt_uav", "ha_snw_pc_uyr", "latitude", "ha_tmp_dc_syr","ha_cly_pc_uav",\
              "area_km2","ha_wet_pc_ug1", "ha_nli_ix_uav", "norm_riverarea", "ha_pnv_cl_smj",\
                  "ha_lit_cl_smj", "soil_SG250_soc_15_30", "soil_Pelletier_sed_thickness",\
                      "ha_lka_pc_sse", "ha_tmp_dc_s06", "ha_tmp_dc_s07", "ha_tmp_dc_s08","ha_tmp_dc_s09"]

r2l=[]
coefl=[]
slopel=[]
inteceptl=[]
p_valuel=[]
namesl=[]
# For each x variable, perform linear regression and get R^2
for x_var in attdf.columns[:-1]:  # Exclude the 'y' column
    if x_var in attl:
        print(x_var)
        X = attdf[x_var].values.reshape(-1, 1) 
        X=X-X.min()+0.01
        X=np.log10(X)
        # Convert to 2D array
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        print(x_var +": {}".format(r2))
        print(model.coef_)
        print("")
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.reshape(42,),y)
        r2l.append(r2)
        coefl.append(model.coef_[0])
        namesl.append(x_var)
        slopel.append(slope)
        inteceptl.append(intercept)
        p_valuel.append(p_value)
sorted_indices = sorted(range(len(r2l)), key=lambda k: r2l[k], reverse=True)
sorted_r2l = [r2l[i] for i in sorted_indices]
sorted_slopel=[slopel[i] for i in sorted_indices]
sorted_inteceptl=[inteceptl[i] for i in sorted_indices]
sorted_p_valuel=[p_valuel[i] for i in sorted_indices]
sorted_names = [namesl[i] for i in sorted_indices]

df_rmse=pd.DataFrame({"Attributes":sorted_names,	"Coefficient":sorted_slopel	,"Intercept":sorted_inteceptl,	"R^2": sorted_r2l,	"p-value": sorted_p_valuel})
df_rmse.to_csv()

plt.plot(np.log(attdf['norm_dis'].values),np.log(attdf['2c_rmse'].values), "*")

text_list = [""]*27
fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(7,4))
ax.bar(range(1,28), sorted_r2l, width=0.8,  align='center',edgecolor="k",color="#61a5c2")
ax.set_xlabel("Basin Attributes", fontsize=13)
ax.set_ylabel("R2", fontsize=13)
ax.set_xticks(range(1,28),text_list )














