# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:41:01 2023

@author: sxc6234
"""

import timeit

start = timeit.default_timer()
from pathlib import Path
from typing import Dict
import random 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.evaluation.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt
import mycolorpy
from mycolorpy import colorlist as mcp
import numpy as np
import contextily as cx
config_file=Path("D:/ShuyuChang/AKTemp/Code/Test_normalized_forcing/Hyperparameter_2.yml")
cudalstm_config = Config(config_file)

# create a new model instance with random weights
cuda_lstm = CudaLSTM(cfg=cudalstm_config)

run_dir = Path("D:/ShuyuChang/AKTemp/Output/Test_normalized_forcing/runs/Test_normalized_forcing_1407_162028/")

# load the trained weights into the new model.
model_path = run_dir / 'model_epoch020.pt'
model_weights = torch.load(str(model_path), map_location='cuda:0')
cuda_lstm.load_state_dict(model_weights)

wshds=['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8']
    
random.seed(10)   
# generate datetime series from '2023-01-01' to '2023-12-31' with daily frequency
date = pd.date_range(start='2002-01-01', end='2022-09-01', freq='D')
date=pd.DataFrame(date)
summer = date[(date[0].dt.month >= 6) & (date[0].dt.month <= 8)]
device = 'cuda:0'



idx_l=[]
for i in range(2): #20 time slots 
    idx= random.choice(list(summer.index.values))
    print(idx)   
    idx_l.append(idx)
    
wshd_stack=[]
for j in range(2):  #10 watersheds 
    wshd=wshds[j]
    print(wshd)
    
    # load the dataset
    scaler = load_scaler(run_dir)
    dataset = get_dataset(cudalstm_config, basin=wshd, is_train=False, period='test', scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    
    
    
    stack = []
    for idx in idx_l:

        counter=0
        for sample in dataloader:
            if counter == (idx):
                break
            counter += 1
        
        
        # compute expected gradients
        n_iter =100 # higher number takes longer but produces better results
        total_eg = []
        
        input_x = sample['x_d'][0,:,:].to(device)
        baseline = torch.from_numpy(np.random.random(size=(input_x.shape))).float().to(device)
        baseline = sample['x_d'][0,:,:].view(len(baseline), 1, 12).float().numpy() # baseline is your input data (7 because I have 7 input variables)
        replace = baseline.shape[0] < n_iter
        sample_idx = np.random.choice(baseline.shape[0], size=(input_x.shape[0], n_iter))
        sampled_baseline = torch.from_numpy(baseline[sample_idx]).float().cuda()
        
        alpha = np.linspace(0, 1, n_iter)
        alpha = torch.from_numpy(alpha).float().to(device)
        alpha = alpha.view(n_iter, *tuple(np.ones(baseline[0].ndim, dtype='int')))

        eg_attributions = []
        for i in range(input_x.shape[0]):
            x = input_x[i].unsqueeze(0)
            ref = sampled_baseline[i]
            scaled_x = ref + alpha * (x - ref)
            attribution = torch.zeros(*scaled_x.shape).to(device)
            for i in range(n_iter):
                part_scaled_x = scaled_x[i:i+1]
                part_scaled_x.requires_grad = True
                s = {'x_d': part_scaled_x.cpu(), 'y': sample['y'], 'date': sample['date'],
                'x_s': sample['x_s']}
                part_y_hat = cuda_lstm(s)['y_hat']
                attribution[i] = torch.autograd.grad(part_y_hat, part_scaled_x)[0]
            integrated = attribution.sum(axis=0) / n_iter
            ig = (x - ref).mean(axis=0) * integrated
            ig = ig.detach().cpu().numpy().squeeze()
            eg_attributions.append(ig)
        eg_attributions = np.array(eg_attributions)
        
        total_eg.append(eg_attributions)
        
        
        stack.append(total_eg[0])
        
    arr_3d = np.stack(stack, axis=-1)
    wshd_stack.append(arr_3d)


arr_4d = np.stack(wshd_stack, axis=-1)


#np.save('D:/ShuyuChang/AKTemp/Data/Expected_gradients/summer_eg_test_1.npy', arr_4d)

#%%

wshds=['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8']
wshds_int=map(int,wshds)  
arr_4d=np.load('D:/ShuyuChang/AKTemp/Data/Expected_gradients/summer_eg_test_lookbackdays_365.npy')
mean_arr_4d=np.mean(np.abs(arr_4d),axis=2)
print(mean_arr_4d.shape)
eg_fi = np.sum(np.abs(mean_arr_4d), axis=0)
print(eg_fi.shape)


varsl = ['snow_depth', 'solar radiation', 'thermal radiation',\
        'air temp', 'wind_u', 'wind_v', 'soil water content','soil_temperature',  "precipitation", "PET", "Runoff","Subsurface Runoff"]

df1 = pd.DataFrame(eg_fi, columns = wshds)
df1["att"]=varsl
df1=df1.set_index("att")
df1_t= df1.T
df1_t["id_int"] =  [int(i) for i in list(df1_t.index.values)] 
df1_t=df1_t.set_index("id_int")
oveall_eg=df1.mean(axis=1)


p="G:/Shared drives/InteRFACE River Temperature (No CUI)/Data/AKTemp_all_atts.csv"    
attdf=pd.read_csv(p,index_col=0)
attdf=attdf[attdf.index.isin(wshds_int) ]  




gdf = gpd.GeoDataFrame( attdf, geometry=gpd.points_from_xy(attdf.longitude, attdf.latitude))
gdf=gdf.set_crs('EPSG:4326')
#gdf = gdf.to_crs('EPSG:5070')

#gdf=gdf.set_index('station_id')
gdf1= gdf.join(df1_t)




#gdf=gdf[~gdf.station_id.isin(["111", "121"])]


#%%

ak=gpd.read_file(r"D:\ShuyuChang\AKTemp\Data\AK_AtlasHCB\AK_AtlasHCB\AK_Historical_Counties\AK_Historical_Counties.shp")
ak=ak.to_crs('EPSG:4326')
ak.geometry=ak.geometry

fig, ax = plt.subplots(1)
 
# Plot the GeoDataFrame without color fill
#ak.boundary.plot(ax=ax, color="black")
colors = mcp.gen_color(cmap="gist_rainbow",n=12)
# Iterate over each polygon (or point)
for _, row in gdf1.iterrows():
    # Get the geometry (Point, LineString, or Polygon)
    geom = row.geometry
    
    # Get the centroid of the geometry
    if geom.geom_type == 'Polygon':
        x, y = geom.centroid.x, geom.centroid.y
    elif geom.geom_type in ['Point', 'LineString']:
        x, y = geom.x, geom.y

    # The data for the pie chart
    sizes = row[varsl]  # replace with your columns

    # Draw a pie chart at each centroid point
    ax.pie(sizes, center=(x, y), colors=colors, radius=1)  # adjust the radius as necessary
    

# Ensure the aspect ratio of the plot is equal
#ax.set_aspect('equal', 'box')



plt.savefig('my_plot.png', transparent=True)

plt.show()
#patches, texts = plt.pie(sizes,  colors=colors, radius=1)
#plt.legend(patches, varsl, bbox_to_anchor=(1, 0, 0.5, 1))
#%%

df2 = pd.DataFrame(eg_fi, columns = wshds)
first=[]
second=[]
third=[]
fourth=[]
fifth=[]   
sixth=[] 
color1=[]
color2=[]
color3=[]
color4=[]
color5=[]
color6=[]

for wshd in list(gdf1.index):
    print(wshd)
    df1_s=df2.sort_values(ascending=False,by=str(wshd))

    
    first.append(list(df1_s.index)[0])
    second.append(list(df1_s.index)[1])
    third.append(list(df1_s.index)[2])
    fourth.append(list(df1_s.index)[3])
    fifth.append(list(df1_s.index)[4])
    sixth.append(list(df1_s.index)[5])
    color1.append(colors[list(df1_s.index)[0]])
    color2.append(colors[list(df1_s.index)[1]])
    color3.append(colors[list(df1_s.index)[2]])
    color4.append(colors[list(df1_s.index)[3]])
    color5.append(colors[list(df1_s.index)[4]])
    color6.append(colors[list(df1_s.index)[5]])
    
gdf1["1"]=first
gdf1["1_color"]=color1
gdf1["2"]=second
gdf1["2_color"]=color2
gdf1["3"]=third
gdf1["3_color"]=color3
gdf1["4"]=fourth
gdf1["4_color"]=color4
gdf1["5"]=fifth
gdf1["5_color"]=color5
gdf1["6"]=sixth
gdf1["6_color"]=color6    
    
gdf2=gdf1.to_crs('EPSG:3857')  
 
for i in range(6):   
    fig, ax = plt.subplots(1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.8)
    gdf2.plot(ax=ax, color=gdf2['{}_color'.format(i+1)] ,edgecolor='black',linewidth = 1,markersize=70)     
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)    
  
varsl = ['snow', 'solar', 'thermal',\
          'air temp', 'wind(u)', 'wind(v)', 'theta','soil temp',  "pcp", "PET", "q","sub q"]
  
for i in range(6):       
    n, bin_edges = np.histogram(gdf2["{}".format(i+1)], 12, range=(0,12))
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    fig, ax=plt.subplots(1, 1, figsize=(8,6))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    ax.bar(bin_middles-0.5, bin_probability, width=bin_width, color=colors, edgecolor='black',linewidth=2) #"#669bbc"
    #ax.set_xlabel("StreamOrder",fontsize=25)
    #ax.set_ylabel("Probability",fontsize=25)
    ax.set_yticks(ticks=[0,0.2,0.4,0.6,0.8],labels=["0","0.2","0.4","0.6","0.8"],fontsize=30)
    ax.set_xticks(range(0,12),labels=["","","","","","","","","","","",""],fontsize=30)
    #ax.set_xticks(range(0,12),labels=varsl,fontsize=30)
    plt.savefig('{}.png'.format(i), format='png', transparent=True)
    plt.show()
    
    







#%%
fig, axs = plt.subplots(6,7, figsize=(15, 14), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)

axs = axs.ravel()
colors = mcp.gen_color(cmap="gist_rainbow",n=12)
for i in range(42):
    
    arr=arr_4d[:,[6,1, 2, 3,5, 7],:,i]
    
    axs[i].plot(range(90), np.mean(arr[0,:,:], axis=-1), color=colors[0])
    axs[i].plot(range(90), np.mean(arr[1,:,:], axis=-1), color=colors[1])
    axs[i].plot(range(90), np.mean(arr[2,:,:], axis=-1), color=colors[2])
    axs[i].plot(range(90), np.mean(arr[3,:,:], axis=-1), color=colors[3])
    axs[i].plot(range(90), np.mean(arr[4,:,:], axis=-1), color=colors[5])
    axs[i].plot(range(90), np.mean(arr[5,:,:], axis=-1) , color=colors[7])        
    axs[i].set_title("wshd = " +wshds[i])
    axs[i].set_ylim(-0.2, 0.5)


fig, axs = plt.subplots(6,7, figsize=(15, 14), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)

axs = axs.ravel()
colors = mcp.gen_color(cmap="gist_rainbow",n=12)
for i in range(42):
    
    arr=arr_4d[:,:,:,i]
    
    for j in range(12):
        axs[i].plot(range(90), np.mean(arr[:,j,:], axis=-1), color=colors[j])
    
    axs[i].set_title("wshd = " +wshds[i])
    axs[i].set_ylim(-0.2, 0.5)


#%%



fig, axs = plt.subplots(1,1, figsize=(8, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)
for axis in ['top','bottom','left','right']:
    axs.spines[axis].set_linewidth(2.5)
colors = mcp.gen_color(cmap="gist_rainbow",n=12)
n1=np.mean(arr_4d,axis=2)
n2=np.mean(n1, axis=-1)    
n3=np.percentile(n1, 75, axis=-1)
n4=np.percentile(n1, 25, axis=-1)

for j in [0,1,2,3,5,7]:
    
    axs.plot(range(365),n2[:,j], color=colors[j],linewidth=3)
    axs.fill_between(range(365),n4[:,j], n3[:,j], where=(n3[:,j] >n4[:,j]), interpolate=True, color=colors[j], alpha=0.3)
#axs.set_ylim(-0.3, 0.4)
#axs.set_yticks(ticks=[-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4],labels=["-0.3","-0.2","-0.1", "0", "0.1", "0.2", "0.3", "0.4"],fontsize=15)
#axs.set_xticks(ticks=[0,30,60,90],labels=["0","30","60", "90"],fontsize=15)




#%%
arr_4d=np.load('D:/ShuyuChang/AKTemp/Data/Expected_gradients/summer_eg_test_lookbackdays_365.npy')
m1=np.mean(np.abs(arr_4d),axis=0)
m2=np.mean(np.abs(m1),axis=1)
varl_s= ['snow', 'solar', 'thermal',\
        'air temp', 'wind_u', 'wind_v', 'soil water','soil temp',  "pcp", "PET", "Runoff","Subsurface Runoff"]
r2= pd.DataFrame(m2.T, columns =varl_s)

r2.describe().loc["50%",:].sort_values(ascending=False)
#%%

t=r2.describe()

'''
"wind_v                0.009183
precipitation         0.019353
wind_u                0.025156
Runoff                0.044703
solar radiation       0.051516
thermal radiation     0.063867
snow_depth            0.065185
PET                   0.103188
Subsurface Runoff     0.217659
soil water content    0.291427
air temp              0.599220
soil_temperature      0.731172"
'''


colorl= [7,3,1,2,0,11,5,10,4,8,9,10]
r2_s=r2[list(t.loc["50%",:].sort_values(ascending=False).index.values)]
varl_s= ['Soil C', 'Air C', 'Soil Water',\
        'Sub Q', 'PET', 'Snow', 'Thermal','Solar',  "Surface Q", "Wind(u)", "Pcp","Wind(v)"]
fig, ax = plt.subplots(figsize=(6,6))



box1=sns.boxplot( x="variable", y="value", data=pd.melt(r2_s), linewidth=2,notch=False,medianprops={"color": "black"},width=0.5, showfliers = False )
i=0
for patch in box1.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor(colors[colorl[i]])
    i+=1




ax.set_ylim([0,0.8])
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
plt.xticks(range(0,12),varl_s,fontsize=20,rotation=30)
#plt.xticks([1],["Test"],fontsize=20)
plt.yticks( fontsize=22)








