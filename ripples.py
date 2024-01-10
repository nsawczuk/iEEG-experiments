import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import scipy


    
    
def find_inpolygon(cClean):
    
##ind_low = find(arrayfun(@(xx) 
#inpolygon(cClean{end,1}(2,1), cClean{end,1}(2,2), cClean{xx,1}(2:end,1), 
#cClean{xx,1}(2:end,2)), [1:length(cClean(1:end-1,1))]),1);
   
    indexs=[]
    for i in range(len(cClean)-1):
        indexs=indexs+[contained_inpolygon(cClean[i][0],[cClean[-1][0][0]])]
    return np.where(indexs)[0]
    
        
def contained_inpolygon(vertices,punto):
    
    #inpolygon
    
    p = path.Path(vertices)
    
    if len(punto)==1:
        return p.contains_points(punto,radius=1e-9)[0]
    else:
        return p.contains_points(punto)
        
def remove_emptylevels(ind,C):
    
    #remove empty levels
    
    aux=[]
    for i in range(len(ind)):
        if len(C[ind[i]])>0:
            aux=aux+[ind[i]]
    
    return aux

def ripples(tC, cdat, hdat, vdat):
##if nargin == 4
## plots = 0;
##end
    Fs = 1000                  ##Sampling Frequency
    P = 2                   ##Group Population Limit

    t = hdat
    tLow = tC - 0.1
    tHigh = tC + 0.1
    fmin = vdat[0]               ## Min Frequency for Analysis
    fmax = vdat[-1]              ## Max Frequency for Analysis
    fLow = 80                 ## Lower Limit Frequency for Event
    fHigh = 200
    fLowR = 70

    det = 0 
    avFreq = 0 
    avMag = 0 
    tStart = 0
    tEnd = 0
    avFreq = 0         
    avMag = 0          
    tDur = 0          
    pkFreq = 0         
    pkMag = 0 
    
    vdat0 = vdat[vdat[:] >= fLow]
    ##hdat0 = hdat(:, (hdat(:,:) >= tLow & hdat(:,:) <= tHigh));
    hdat0=hdat[hdat[:]>=tLow]
    hdat0=hdat0[hdat0[:]<=tHigh]
    #cdat0 = cdat(ismember(vdat,vdat0),ismember(hdat,hdat0));
    cdat0= cdat[:,np.isin(hdat,hdat0)][np.isin(vdat,vdat0)]
    levels = np.linspace(np.min(cdat0), np.max(cdat0), 52)
    levels = levels[range(1,51)]
    #C = contourc(hdat0, vdat0, cdat0, levels)';
    plt.ioff()
    C= plt.contour(hdat0,vdat0,cdat0,levels).allsegs
    plt.clf()    
        
## FIND ALL CONTOURS ABOVE THRESHOLD AND GROUP CLOSED LOOP CONTOURS
    thresh = 0.2*np.max(cdat0)         #% Set threshold to 20% of max power
    ind= np.where(levels>thresh)[0]
    ind= remove_emptylevels(ind,C)
    
    olc = []
    results = []
    
    if len(ind)==0 or thresh < 1 : ##or size(find(C(:) > thresh),1) > size(C,1):
        results = []
    else:
    ##clsd_lp: FIND ALL CLOSED LOOP CONTOURS ABOVE THRESHOLD
        clsd_lp = [] 
        for ee in range(len(ind)):
            
            for j in range(len(C[ind[ee]])):
                if np.array_equal(C[ind[ee]][j][0],C[ind[ee]][j][-1]):
                    clsd_lp.append([C[ind[ee]][j]])
                else:
                    if np.array_equal([C[ind[ee]][j][0][1],C[ind[ee]][j][-1][1]], [fmax,fmax]):        #% (#1)
                        olc.append([C[ind[ee]][j]])                   #% (#1)
 

    ##% GROUP CLOSED-LOOP CONTOURS 
        cClean = clsd_lp.copy()
        clp_grps = []
        grp_ind = -1
        clcGrp = []
        cGrp = []

        while len(cClean) !=0:
            ind_low= find_inpolygon(cClean)
            if len(ind_low)>0: 
                grp_ind = grp_ind + 1      # Increment Group Index
                clp_grps.append([])
                clp_grps[grp_ind].append(cClean[-1][0])
                cClean.pop()
                for jj in range(len(cClean)-1,ind_low[0]-1,-1):
                    #if inpolygon(cClean{jj,1}(2,1), cClean{jj,1}(2,2), cClean{ind_low,1}(2:end,1), cClean{ind_low,1}(2:end,2))
                     if contained_inpolygon(cClean[ind_low[0]][0],[cClean[jj][0][0]]):   
                        clp_grps[grp_ind].append(cClean[jj][0])
                        cClean.pop(jj)
                
            else:
                cClean.pop()

    
        if len(clsd_lp) == 0:
            det = 0 
            avFreq = 0 
            avMag = 0 
            tStart = 0
            tEnd = 0

            avFreq = 0         
            avMag = 0          
            tDur = 0          
            pkFreq = 0         
            pkMag = 0          

        elif len(clp_grps) == 0:
            det = 0 
            avFreq = 0 
            avMag = 0 
            tStart = 0
            tEnd = 0

            avFreq = 0         
            avMag = 0          
            tDur = 0          
            pkFreq = 0         
            pkMag = 0
       

        elif len(clp_grps) >0:

        #% Calculate Frequency and Time Weighted Averages for All Remaining Groups
            cGrp = []
            for jj in range (len(clp_grps)):
                cGrp.append([clp_grps[jj],jj])
            
            H, V = np.meshgrid(hdat0,vdat0)
            HV = [H,V]
            HV=np.append(HV[0].reshape(-1,1),HV[1].reshape(-1,1),axis=1)
            perim = []
            inLg = []
            freq_wm = []
            freqRef = []
            time_wm = []
            pow_m = []
            powerRef = []

            cgrpF = []         
            cdatGrp = []       
            pow_pk = []        
            time_pk = []       
            freq_pk = []
            
            for jj in range(len(clp_grps)-1,-1,-1):
                perim.append(clp_grps[jj][-1])
                inLg.append(contained_inpolygon(perim[-1],HV).reshape(H.shape))
                
                freq_wm.append((np.sum(V[inLg[-1]]*cdat0[inLg[-1]]))/np.sum(cdat0[inLg[-1]]))
                freqRef.append(freq_wm[-1])
                time_wm.append((np.sum(H[inLg[-1]]*cdat0[inLg[-1]]))/np.sum(cdat0[inLg[-1]]))
                pow_m.append(np.mean(cdat0[inLg[-1]]))                                       
                powerRef.append(pow_m[-1])

                cdatGrp.append(cdat0*inLg[-1])
                
                pow_pk.append(np.max(cdatGrp[-1]))
                iPk = np.unravel_index(np.argmax(cdatGrp[-1]),cdatGrp[-1].shape)
                time_pk.append(H[iPk])
                freq_pk.append(V[iPk]) 

            # REMOVE GROUPS WITH MEAN TIME > 30 ms MINIMUM TIME > 10 ms
            # FROM 0.5 MARK OR IF MEAN FREQ ABOVE 200 OR IF GROUP
            # CONTAINS <= P CONTOURS
                if freq_wm[-1] > fHigh or len(cGrp[jj][0]) <= P:
                    cGrp.pop(jj)
            
            perim.reverse()
            inLg.reverse()
            freq_wm.reverse()
            freqRef.reverse()
            time_wm.reverse()
            pow_m.reverse()
            powerRef.reverse()

            cgrpF.reverse()
            cdatGrp.reverse()
            pow_pk.reverse()
            time_pk.reverse()
            freq_pk.reverse()
            
            if len(cGrp) == 0 :   #% If there are no groups within time constraints
                det = 0
                avFreq = 0 
                avMag = 0 
                tStart = 0
                tEnd = 0
                tDur = 0           
                pkFreq = 0         
                pkMag = 0          

            else: #    % if there is 1 group
                
                cgrpF=[]
                
                for I in range(len(cGrp)):
                    det = 1 
                    avFreq = freqRef[cGrp[I][1]]
                    avMag = powerRef[cGrp[I][1]]
                    tStart = np.min(cGrp[I][0][-1][:,0]) 
                    tEnd = np.max(cGrp[I][0][-1][:,0])
                    cgrpF.append(cGrp[I])              
                    tDur = tEnd - tStart           
                    pkFreq = freq_pk[cGrp[I][1]]  
                    pkMag = pow_pk[cGrp[I][1]]
                    results.append([det,avFreq,pkFreq,avMag,pkMag,tDur,tStart,tEnd])

#% REFLEX TESTING AT 80HZ LOWER LIMIT
    if len(results) != 0:
        
        for I in range(len(results)):
            if np.min(cgrpF[I][0][-1][:,1]) <= fLow+1:
                xpk = time_pk[cgrpF[I][1]]
                ypk = freq_pk[cgrpF[I][1]]
        
                vdat0 = vdat[vdat[:] >= fLowR]
                hdat0=hdat[hdat[:]>=tLow]
                hdat0=hdat0[hdat0[:]<=tHigh]    
                cdat0= cdat[:,np.isin(hdat,hdat0)][np.isin(vdat,vdat0)]
                C= plt.contour(hdat0,vdat0,cdat0,levels).allsegs
                ind= np.where(levels>thresh)[0]
                ind= remove_emptylevels(ind,C)            

                clsd_lp = []   

                for ee in range(len(ind)):
                    for j in range(len(C[ind[ee]])):
                        if np.array_equal(C[ind[ee]][j][0],C[ind[ee]][j][-1]):
                            clsd_lp.append(C[ind[ee]][j])

                cGrp[0][0]=[]

                for jj in range(len(clsd_lp)):
                    if contained_inpolygon(clsd_lp[jj],[[xpk,ypk]]):
                        cGrp[0][0].insert(0,clsd_lp[jj])

                H, V = np.meshgrid(hdat0,vdat0)
                HV = [H,V]
                HV=np.append(HV[0].reshape(-1,1),HV[1].reshape(-1,1),axis=1)            
                perim =cGrp[0][0][-1]
                inLg = contained_inpolygon(perim,HV).reshape(H.shape)
                
                results[I][1]= np.sum(V[inLg]*cdat0[inLg])/np.sum(cdat0[inLg])# 2  Weighted Mean Frequency (Reflex 80)
                
                results[I][3]= np.mean(cdat0[inLg])                                #% 4  Mean Power  (Reflex 80)
                results[I][5]= np.max(perim[:,0]) - np.min(perim[:,0])               #% 6  tDuration  (Reflex 80)
                
                
            
#results =[det,avFreq,pkFreq,avMag,pkMag,tDur,tStart,tEnd]
 # REFLEX TESTING AT 200 HZ
    #if len(olc)!=0:
     #   results.append([0,0,0,0,0,0,0,0])
        
   
        
    return results

