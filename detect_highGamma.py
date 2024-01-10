import scipy.io
import numpy as np
import pandas as pd
import time
import sys
import mne
import mat73
from wavelet import wavelets
from ripples import ripples



def get_tag(BP,row):
    return BP['tagName'][row['bp']]
def get_coordinates(BP,row):
    return BP['Electrode coordinates'][row['bp']]
def get_eType(BP,row):
    if (BP['eType'][row['bp']][0]=='D'):
        return 1
    else:
        return 0

def hg_ripples_3s(eeg, cdat, hdat, vdat,bp,k,length):
    #detects all ripples in a one second length clip
    times= np.linspace(0,int(length/1000), int(length/100), endpoint=False)[1:]
    results=[]
    for t in times:
        res=ripples(t,cdat, hdat, vdat)
        if len(res)!=0:
            for j in range(len(res)):
                res[j][6]=res[j][6]
                res[j][7]=res[j][7]
                res[j].insert(0,k)
                res[j].insert(0,bp)
            results=results+res
    return results
    
def hg_ripples(eeg,wavelet, freqoi, timeoi,tapM,k,length):
    #detects all ripples
    results=[]

    transform_power=np.abs(mne.time_frequency.tfr.cwt(eeg, wavelet))**2
    
    for i in range(transform_power.shape[0]):
    	transform_power[i,:,:] = transform_power[i,:,:]*tapM
    	transform_power[i,:,:] = transform_power[i,:,:] -np.mean(transform_power[i,:,:],axis=1)[:,None]
            
    for i in range(transform_power.shape[0]):
        results= results+hg_ripples_3s(eeg, transform_power[i,:,:], timeoi, freqoi,i,k,length)
    

    return results


def highgamma(eeg,wavelet, freqoi, timeoi,tapM,k,length):
   
    results= hg_ripples(eeg, wavelet, freqoi, timeoi,tapM, k,length)
    column=["bp","trial","det","avFreq","pkFreq","avMag","pkMag","tDur","tStart","tEnd"]
    res= np.array(results)
    res= pd.DataFrame(res, columns=column)

    res=res.sort_values('tStart')
    res['diff'] = res.groupby('bp')['tStart'].diff()
    res= res.fillna(1)
    res=res[res['diff'].ge(0.005)].drop('diff',axis=1)

    dataframeFinal= pd.DataFrame(columns=['bp', 'trial', 'det', 'avFreq', 'pkFreq', 'avMag','pkMag', 'tDur', 'tStart', 'tEnd', 'rounded_pkFreq'])
    DF=[]
    BPx= res['bp'].unique()
    res['rounded_pkFreq']=round(res['pkFreq'])
    for i in BPx:
        datosaux=res[res['bp']==i]
        freqs= datosaux['rounded_pkFreq'].unique()
        for f in freqs:
            datfreq= datosaux[datosaux['rounded_pkFreq']==f]
            a = np.triu(datfreq['tEnd'].values > datfreq['tStart'].values[:, None])
            b = np.triu(datfreq['tStart'].values < datfreq['tEnd'].values[:, None])
            DF.append(datfreq[(a & b).sum(0) == 1])
            dataframeFinal=pd.concat([dataframeFinal,datfreq[(a & b).sum(0) == 1]])

    dataframeFinal= pd.concat(DF)
    dataframeFinal=dataframeFinal.drop(['rounded_pkFreq','det'],axis=1)

    return res
    
    
def detect_hg(subject):

    
    if 'RAM' in subject:
        T= scipy.io.loadmat(subject) 
        y= np.array(pd.DataFrame(T['Words'][0])['recalled'].astype(int))
        BP=pd.DataFrame(T['bpElec2'][0])
        BIPOLAR= T['BIPOLAR2'][0]
        WORDS=pd.DataFrame(T['Words'][0])['recalled']
        filename=subject[6:-5] 
        b=T['BIPOLAR2'][0][0].shape[1] #last trial
        BP['x']=BP['x'].astype(float)
        BP['y']=BP['y'].astype(float)
        BP['z']=BP['z'].astype(float)
        BP["Electrode coordinates"] = BP[["x","y","z"]].apply(tuple, axis=1)            
    else:
        
        T= mat73.loadmat(subject) 
        y= np.array(pd.DataFrame(T['Words'])['recalled'].astype(int))

        BIPOLAR= T['BIPOLAR']
        if len(BIPOLAR)>1:
            BP=pd.DataFrame(T['bpElec'])        
            BP['x']=BP['x_tal'].astype(float)
            BP['y']=BP['y_tal'].astype(float)
            BP['z']=BP['z_tal'].astype(float)
            BP["Electrode coordinates"] = BP[["x","y","z"]].apply(tuple, axis=1)
        else:
            BP=T['bpElec']
            BP['x']=BP['x_tal'].astype(float)
            BP['y']=BP['y_tal'].astype(float)
            BP['z']=BP['z_tal'].astype(float)
            BP["Electrode coordinates"] = (float(BP['x']),float(BP['y']),float(BP['z']))
            
        WORDS=pd.DataFrame(T['Words'])['recalled']    
        filename=subject[6:-5]
        b=T['BIPOLAR'][0][0].shape[0] #last trial

        
        
    freqR=[50,250]
    length= BIPOLAR[0].shape[0]
    
    width=7
    gwidth=5
    wavelet, freqoi, timeoi, tapM= wavelets(length,freqR,width,gwidth)


    a=0

    #DF= pd.DataFrame(columns=['bp', 'trial', 'det', 'avFreq', 'pkFreq', 'avMag','pkMag', 'tDur', 'tStart', 'tEnd'])
    aux=[]
    for x in range(a,b):      
        eeg=BIPOLAR[0][:,x]
        eeg=eeg.reshape((1,length))
        for i in range(1,len(BIPOLAR)):
            eeg1=BIPOLAR[i][:,x]
            eeg1=eeg1.reshape((1,length))
            eeg=np.concatenate((eeg,eeg1))

        Res=highgamma(eeg,wavelet, freqoi, timeoi,tapM,x,length)
        if 'RAM' in subject:
        	Res['Recalled']=WORDS[x][0][0]
        else:
        	Res['Recalled']=WORDS[x]
        Res['High-gamma/Beta']=0
        
        
        if not('RAM' in subject) and len(BIPOLAR)==1:
        
            Res['Electrode name'] = BP['tagName']
            Res['Electrode coordinates'] = str(BP['Electrode coordinates'])
            Res['Electrode subdural/depth contact'] = int(BP['eType'][0]=='D')
        
        else:    
        
            Res['Electrode name'] = Res.apply(lambda row: get_tag(BP,row), axis=1)
            Res['Electrode coordinates'] = Res.apply(lambda row: get_coordinates(BP,row), axis=1)
            Res['Electrode subdural/depth contact'] = Res.apply(lambda row: get_eType(BP,row), axis=1)
            
        Res=Res.drop(['bp'],axis=1)

        aux.append(Res)

    DF=pd.concat(aux).reset_index(drop=True)
    DF=DF.sort_values('Electrode name')
    DF.to_csv('highgamma/'+filename+"hg.csv")

    
detect_hg(sys.argv[1])
