# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:34:13 2024

@author: sxc6234
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

#load IG 
eg_arr = np.load('./runs/Final_20230921_2909_103637/ig.npy')
#define catchments for further analysis

wshds = ['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8']
    
#define forcings for further analysis
varsl = ['snow_depth', 'solar radiation', 'thermal radiation',\
        'air temp', 'wind_u', 'wind_v', 'soil water content','soil_temperature',  "precipitation", "PET", "Runoff","Subsurface Runoff"]
#%%
#the time- and basin-averaged integrated gradients as a function of lag time (days before present) for the top with the 25th and 75th quantile confidence intervals.

colors = ["#d73027","#fc8d59", "#fee090", "#e0f3f8", "#af8dc3", "#af8dc3", "#91bfdb", "#4575b4", "#af8dc3", "#af8dc3", "#af8dc3", "#af8dc3"]
colors = ["#c51b7d", "#e9a3c9", "#fde0ef", "#e6f5d0", "#fddbc7","#fddbc7","#a1d76a", "#4d9221","#fddbc7","#fddbc7","#fddbc7","#fddbc7",]
colorsv = mcp.gen_color(cmap="viridis",n=6)
colors = ["#c86bfa","#1b4965", "#bee9e8", "#2c6e49",  "#fddbc7","#fddbc7","#5c0099", colorsv[4], "#fddbc7","#fddbc7","#fddbc7","#fddbc7"]
n1 = np.mean(eg_arr,axis=2) #days
n2 = np.mean(n1, axis=1)    
n3 = np.percentile(n1, 75, axis=1)
n4 = np.percentile(n1, 25, axis=1)
n1_r = n1[ ::-1,:]
n2_r = n2[ ::-1,:]
n3_r = n3[ ::-1,:]
n4_r = n4[ ::-1,:]


fig, axs = plt.subplots(3, 4, figsize=(15, 8), facecolor = 'w', edgecolor = 'k')
fig.subplots_adjust(hspace = .5, wspace =.5)
l=0
for ax in axs.flatten():

    ax.plot(range(365), n2_r[:,l], color=colors[l], linewidth=2.5, linestyle='-', marker='s', markersize=4)
    
    ax.fill_between(range(365),n4_r[:,l], n3_r[:,l], interpolate=False, color=colors[l], alpha=0.5,linewidth=0)
    ax.set_xlim([0,15])
    ax.set_title(varsl[l])
    ax.axhline(y=0, color='k', linestyle='--')
    l+=1
#plt.savefig(r"D:\ShuyuChang\AKTemp\Figrues\fig71.png", dpi=300)  


#%%
#Boxplot of time-averaged, absolute sum of integrated gradients for different basins (absolute values)
n1=np.mean(eg_arr,axis=2) #365*42*12
n1=np.abs(n1)
n2=np.sum(n1,axis=0)

n2_m=np.median(n2,axis=0)

descending_indices = np.argsort(n2_m)[::-1]
n2 = n2[:, descending_indices ]
varsl_r= [varsl[i] for i in descending_indices]
colors_r=[colors[i] for i in descending_indices]


n2_df=pd.DataFrame(n2)
df_percent = n2_df.div(n2_df.sum(axis=1), axis=0) * 100

fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    axs.spines[axis].set_linewidth(1)

box=axs.boxplot(n2,showfliers=False,notch=False, widths=0.5, patch_artist=True )
axs.set_yticks([0,0.2,0.4,0.6,0.8],["0","0.2","0.4","0.6","0.8"],fontsize=15)
axs.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12],["1","2","3","4","5","6","7","8","9","10","11","12"],fontsize=15)
for patch, color in zip(box['boxes'], colors_r):
    patch.set_facecolor(color)
for element in ['boxes', 'whiskers', 'caps', ]:
    plt.setp(box[element], linewidth=1.0)
plt.setp(box['medians'], color='black', linewidth=1.0)

axs.set_xlabel("")
axs.set_ylabel("")

plt.title('')

##Boxplot of time-averaged, absolute sum of integrated gradients for different basins (percentage)
fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    axs.spines[axis].set_linewidth(1)


box=axs.boxplot(df_percent,showfliers=False,notch=False, widths=0.5, patch_artist=True )
for patch, color in zip(box['boxes'], colors_r):
    patch.set_facecolor(color)
for element in ['boxes', 'whiskers', 'caps', ]:
    plt.setp(box[element], linewidth=1.0)
plt.setp(box['medians'], color='black', linewidth=1.0)

#%%
#The time- and basin-averaged cumulative contribution percentage for the top six forcing as a function of lag time.

eg_arr_abs=np.abs(eg_arr)
n=np.mean(eg_arr_abs,axis=2)
n=np.abs(n)
n_sum=np.sum(n,axis=-1)
n_sum=np.sum(n_sum,axis=0)
cum_per=np.zeros((365,6))
fig, axs = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
i=0
for l in descending_indices[0:6]:

    ns=n[:,:,l]
    ns=ns[ ::-1,:]
    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(ns,axis=0)
    
    # Convert cumulative sum to percentage of the total sum
    total_sum = np.sum(np.sum(ns))
    cumulative_percentage = 100 * cumulative_sum / n_sum
    
    n1=np.percentile(cumulative_percentage, 50, axis=1)
    n2=np.percentile(cumulative_percentage, 75, axis=1)
    n3=np.percentile(cumulative_percentage, 25, axis=1)
    axs.plot(range(365),n1, color=colors[l],linewidth=3,linestyle='--')#, marker='s', markersize=2
    #axs.plot(range(365),n2, color=colors[l],linewidth=1,linestyle='-', alpha=0.8)
    #axs.plot(range(365),n3, color=colors[l],linewidth=1,linestyle='-', alpha=0.8)
    axs.fill_between(range(365),n2, n3, interpolate=False, color=colors[l], alpha=0.8,linewidth=0)
    
axs.set_xticks([0,25,50,75,100],["0","25","50","75","100"],fontsize=15)
axs.set_yticks([0,10,20,30],["0","10","20","30"],fontsize=15)
axs.set_xlabel("")
axs.set_ylabel("")

   
axs.set_xlim(0,100)
axs.set_ylim(0,33)
# Optionally, set a general title
plt.title('')
#%%
#Boxplots showing the number of days prior to the current prediction when the cumulative contribution of each forcing variable exceeds specific thresholds

# Create the initial array of numbers 1 to 5
threshold=[50,70,90,99]
threshold_array = np.array(range(1,(len(threshold)+1)))
repeated_threshold= np.repeat(threshold_array, 42)
repeated_threshold = np.tile(repeated_threshold, 6)
initial_f = np.array([1, 2, 3, 4, 5,6])
repeated_f= np.repeat(initial_f, 42*len(threshold))


n=np.mean(eg_arr_abs,axis=2)

days_sum=np.zeros((1,1))
for l in descending_indices[0:6]:
    ns=n[:,:,l]
    ns=ns[ ::-1,:]

    cumulative_sum = np.cumsum(ns,axis=0)
    
    # Convert cumulative sum to percentage of the total sum
    total_sum = np.sum(ns,axis=0)
    cumulative_percentage = 100 * cumulative_sum / total_sum
    days=np.zeros(shape=(42,len(threshold)))
    for t in range(len(threshold)):

        indexes = np.argmax(cumulative_percentage>= threshold[t], axis=0)
        days[:,t]=indexes
        
    days_sum = np.vstack((days_sum, days.flatten(order='F').reshape(len(days.flatten()),1)))
    
palette = {1: colors_r[0], 2: colors_r[1],3: colors_r[2], 4: colors_r[3],5: colors_r[4], 6: colors_r[5]}    
df=pd.DataFrame({"forcing":repeated_f.flatten(), "threshold":repeated_threshold.flatten(), "days":days_sum[1:,].flatten()})    

fig, ax = plt.subplots(figsize=(6,6))
box=sns.boxplot(x="threshold", hue="forcing", y="days", data=df, ax=ax, width=0.7, showfliers=False,\
                palette=palette,saturation=1,linewidth=1.0,\
                    medianprops={"color": "k", "linewidth": 1.2})


ax.legend_.remove()
ax.set_xticks([0,1,2,3],["50%","70%","90%","99%"],fontsize=15)
ax.set_yticks([0,50,100,150,180],["0","50","100","150","180"],fontsize=15)
ax.set_xlabel("")
ax.set_ylabel("")

#plt.savefig(r"D:\ShuyuChang\AKTemp\Figrues\fig8.png", dpi=300)  
dfs=df[df["threshold"]==3]

for i in range(1,7):
    dfss=dfs[dfs["forcing"]==i]
    print(dfss.describe())

#%%
eg_arr_abs=np.abs(eg_arr)
n=np.mean(eg_arr_abs,axis=2)
n=np.abs(n)

#cumulative attribution of each forcing for each wshd 
att_wshd=np.zeros((42,6))

for w in range(42):

    
    cum_per=np.zeros((365,6))

  
    ns=n[:,w,:]#365*12

    
        #ns=n[:,:,l]
    ns=ns[ ::-1,:]
    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(ns,axis=0)
    
    # Convert cumulative sum to percentage of the total sum
    total_sum = np.sum(np.sum(ns))
    cumulative_percentage = 100 * cumulative_sum / total_sum
    att_wshd[w,:]=cumulative_percentage[-1,list(descending_indices[0:6])]
    
bix=[]
snow=[]
for wshd in wshds:
    fdf=pd.read_csv("./Data/Rawdata/forcings/"+wshd+".csv")
    fdf_s=fdf["sub_surface_runoff_sum"].sum()/(fdf["sub_surface_runoff_sum"].sum()+fdf["surface_runoff_sum"].sum())
    bix.append(fdf_s)
    snw=fdf['snow_depth_water_equivalent_mean'].mean()
    snow.append(snw)

#%% spatial analysis
rep=["topo_GM90_topo_idx",\
"ha_ppd_pk_uav",\
"ha_for_pc_use",\
"ha_snd_pc_uav",\
"ice_PastickAK_probability",\
"norm_aet",\
"ice_GLIMS_glacier",\
"ha_cmi_ix_uyr",\
"norm_dis",\
"ha_ele_mt_uav",\
"ha_snw_pc_uyr",\
"latitude",\
"ha_tmp_dc_syr",\
"ha_cly_pc_uav",\
"norm_inundation",\
"area_km2",\
"ha_wet_pc_ug1",\
"ha_nli_ix_uav",\
"ha_pnv_cl_smj",\
"ha_lit_cl_smj",\
"soil_SG250_soc_15_30",\
"soil_Pelletier_sed_thickness",\
"ha_dis_m3_pyr",\
"ha_ari_ix_uav",\
"station_id"]
    
f=["airt","soilt","solar","thermal","theta","sw"]
for i in range(1,5):
    dfs=df[df["threshold"]==i]
    dfs=dfs[dfs["forcing"]==1]
    dfs["wshd"]=wshds
    print(dfs.sort_values('days'))

int_wshd=[int(x) for x in wshds]
att=pd.read_csv("./Data/Rawdata/AKTemp_all_atts.csv")
att=att[rep]
att=att[att["station_id"].isin(int_wshd)]
att['station_id'] = att['station_id'].astype(str)
att.set_index('station_id', inplace=True)
att = att.reindex(wshds)


for i in range(1,5):
    dfs=df[df["threshold"]==i]
    for j in range(1,7):
        dfss=dfs[dfs["forcing"]==j]
        dfss["wshd"]=wshds
        att[f[j-1]+"_{}%".format(threshold[i-1])]=dfss["days"].values



att["airt_cum_att"]=att_wshd[:,0]
att["soilt_cum_att"]=att_wshd[:,1]
att["theta_cum_att"]=att_wshd[:,-2]
att["snow_cum_att"]=att_wshd[:,-1]
att["snow_mean"]=snow

#%%
#The relationships  between the impacts (cumulative attribution (%)) of soil temperature (green) and SWE (purple)  and mean snow water depth, basin glacier coverage , and mean discharge

def lr(df, x_col, y_col):
    """
    Perform linear regression between two DataFrame columns and calculate Pearson correlation coefficient and p-value.

    Parameters:
        df (DataFrame): The DataFrame containing the columns.
        x_col (str): The name of the independent variable column.
        y_col (str): The name of the dependent variable column.

    Returns:
        tuple: A tuple containing the slope, intercept, Pearson correlation coefficient, and p-value.
    """
    # Extract the columns as numpy arrays
    x = df[x_col].values
    y = df[y_col].values

    # Perform linear regression
    slope, intercept = np.polyfit(x, y, 1)

    # Calculate Pearson correlation coefficient and p-value
    pearson_corr, p_value = pearsonr(x, y)
    
    print(pearson_corr,p_value)

    return slope, intercept, pearson_corr, p_value
slope1, intercept1, pearson_corr1, p_value1=lr(att,"snow_mean","snow_cum_att")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"snow_mean","soilt_cum_att")
x_values = np.linspace(0, 1200,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["snow_mean"],att["snow_cum_att"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["snow_mean"],att["soilt_cum_att"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(0,500)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')


slope1, intercept1, pearson_corr1, p_value1=lr(att,"ice_GLIMS_glacier","snow_cum_att")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"ice_GLIMS_glacier", "soilt_cum_att")
x_values = np.linspace(0, 1,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["ice_GLIMS_glacier"],att["snow_cum_att"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["ice_GLIMS_glacier"],att["soilt_cum_att"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(0,0.01)
ax1.set_ylim(0,40)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')

slope1, intercept1, pearson_corr1, p_value1=lr(att,"ha_dis_m3_pyr","snow_cum_att")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"ha_dis_m3_pyr", "soilt_cum_att")
x_values = np.linspace(0, 100,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["ha_dis_m3_pyr"],att["snow_cum_att"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["ha_dis_m3_pyr"],att["soilt_cum_att"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(0,100)
#ax1.set_ylim(0,40)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')


#%%%
#Cumulative contribution percentage of the top six forcings as a function of lag time (days before present) for three representative basins from the study, which are denoted by basin number
eg_arr_abs=np.abs(eg_arr)
n=np.mean(eg_arr_abs,axis=2)
n=np.abs(n)
fig, axs = plt.subplots(6,7, figsize=(23,15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .4, wspace=.2)
#cumulative attribution of each forcing for each wshd 
att_wshd=np.zeros((42,6))
w=0
for ax in axs.flatten():
    
    cum_per=np.zeros((365,6))

    i=0
    ns=n[:,w,:]#365*12

    
        #ns=n[:,:,l]
    ns=ns[ ::-1,:]
    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(ns,axis=0)
    
    # Convert cumulative sum to percentage of the total sum
    total_sum = np.sum(np.sum(ns))
    cumulative_percentage = 100 * cumulative_sum / total_sum
    att_wshd[w,:]=cumulative_percentage[-1,list(descending_indices[0:6])]
    
    for l in descending_indices[:6]:

        ax.plot(range(365),cumulative_percentage[:,l], color=colors[l],linewidth=2,linestyle='-')#, marker='s', markersize=2

    ax.set_ylim(0,38)
    ax.set_xlim(0,50)
    ax.set_title(wshds[w])
    w+=1

#plt.savefig(r"D:\ShuyuChang\AKTemp\Figrues\fig82.png", dpi=300)  
# %%
# Relationships between lag time (days before present) of soil temperature and SWE, and (b) soil organic content, (c) sand percentage, and (d) climate index

slope1, intercept1, pearson_corr1, p_value1=lr(att,"soil_SG250_soc_15_30","soilt_90%")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"soil_SG250_soc_15_30","sw_90%")
x_values = np.linspace(500, 1800,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["soil_SG250_soc_15_30"],att["soilt_90%"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["soil_SG250_soc_15_30"],att["sw_90%"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(500,1800)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')


slope1, intercept1, pearson_corr1, p_value1=lr(att,"ha_snd_pc_uav","soilt_90%")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"ha_snd_pc_uav", "sw_90%")
x_values = np.linspace(30,70,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["ha_snd_pc_uav"],att["soilt_90%"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["ha_snd_pc_uav"],att["sw_90%"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(30,70)
ax1.set_ylim(0,100)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')

slope1, intercept1, pearson_corr1, p_value1=lr(att,"ha_cmi_ix_uyr","soilt_90%")
slope2, intercept2, pearson_corr2, p_value2=lr(att,"ha_cmi_ix_uyr", "sw_90%")
x_values = np.linspace(-60,100,50)  # 100 points between 0 and 10

# Calculate corresponding y-values
y1= slope1 * x_values + intercept1
y2= slope2 * x_values + intercept2


fig, ax1 = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(1)
ax1.plot(att["ha_cmi_ix_uyr"],att["soilt_90%"],marker='o', markersize=18, color=colors_r[5], linewidth=0,alpha=0.4)
ax1.plot(x_values,y1,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.plot(att["ha_cmi_ix_uyr"],att["sw_90%"],marker='o', markersize=18, color=colors_r[1], linewidth=0,alpha=0.4)
ax1.plot(x_values,y2,linewidth=5, color="k",linestyle="--",alpha=0.7)
ax1.set_xlim(-60,100)
#ax1.set_ylim(0,40)
plt.xticks(fontsize=18, fontfamily='sans-serif')
plt.yticks(fontsize=18, fontfamily='sans-serif')

   