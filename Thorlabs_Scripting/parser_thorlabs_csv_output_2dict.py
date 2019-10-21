import pandas as pd
import os
import datetime
import json

# change dir not working
#os.chdir(r'P:\gsg\gsgstud\07_Interdisciplinary_Project\IPA_Meyer_Weiss\99_Git\NeuralNet\data_191014')

pmma_02 = {'0':{'reference':dict()}}
pmma_03 = {'0':{'reference':dict()}}
pmma_04 =  {'0':{'reference':dict()}}
pmma_05 =  {'0':{'reference':dict()}}
pc_01 =  {'0':{'reference':dict()}}
pc_02 =  {'0':{'reference':dict()}}
pc_03 =  {'0':{'reference':dict()}}
pc_04 =  {'0':{'reference':dict()}}
pc_05 =  {'0':{'reference':dict()}}
raw =  {'0':{'reference':dict()}}
epoch = 0

for filename in os.listdir('data_191014'):
    fileloc = os.path.join('data_191014',filename)
    df = pd.read_csv(fileloc,sep=';',header=None,skiprows=[0,1,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32])
    
    # creating dict per measurement
    date = int(df.iloc[0].values[1])
    time = int(df.iloc[1].values[1]/100)
    key = str(date)+'-'+str(time)
    integration_time = int(df.iloc[2].values[1]) # ms
    wav = df.iloc[4:-1,0].values
    wav = wav.astype(float)
    meas = df.iloc[4:-1,1].values

    if filename[0:2]=='PC':
        if filename[3:13]=='spectralon':
            spec = filename[-23:-19] + '_' + filename[-18:-16]
            if filename[-6:-4]=='01':
                pc_01['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='02':
                pc_02['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='03':
                pc_03['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='04':
                pc_04['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='05':
                pc_05['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
        elif filename[9:11]=='01':
            pc_01['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[9:11]=='02':
            pc_02['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[9:11]=='03':
            pc_03['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[9:11]=='04':
            pc_04['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[9:11]=='05':
            pc_05['0'][key] = (wav.tolist(),meas.tolist(),integration_time)

    if filename[0:4]=='PMMA':
        if filename[5:15]=='spectralon':
            if filename[-6:-4]=='02':
                pmma_02['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='03':
                pmma_03['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='04':
                pmma_04['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
            elif filename[-6:-4]=='05':
                pmma_05['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
        elif filename[11:13]=='02':
            pmma_02['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[11:13]=='03':
            pmma_03['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[11:13]=='04':
            pmma_04['0'][key] = (wav.tolist(),meas.tolist(),integration_time)
        elif filename[11:13]=='05':
            pmma_05['0'][key] = (wav.tolist(),meas.tolist(),integration_time)

    if filename[0:3]=='raw':
        if filename[4:14]=='spectralon':
                raw['0']['reference'][key] = (wav.tolist(),meas.tolist(),integration_time,spec)
        else:
            raw['0'][key] = (wav.tolist(),meas.tolist(),integration_time)

with open('data\pmma_02.json', 'w') as fp:
    json.dump(pmma_02, fp)
with open('data\pmma_03.json', 'w') as fp:
    json.dump(pmma_03, fp)
with open('data\pmma_04.json', 'w') as fp:
    json.dump(pmma_04, fp)
with open('data\pmma_05.json', 'w') as fp:
    json.dump(pmma_05, fp)
with open('data\pc_01.json', 'w') as fp:
    json.dump(pc_01, fp)
with open('data\pc_02.json', 'w') as fp:
    json.dump(pc_02, fp)
with open('data\pc_03.json', 'w') as fp:
    json.dump(pc_03, fp)
with open('data\pc_04.json', 'w') as fp:
    json.dump(pc_04, fp)
with open('data\pc_05.json', 'w') as fp:
    json.dump(pc_05, fp)
localisa = os.path.join('data','background.json')
with open(localisa, 'w') as fp:
    json.dump(raw, fp)