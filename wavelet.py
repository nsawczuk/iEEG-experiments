import numpy as np
import math


def wavelets(N,freq,width,gwidth):
    
    Fs = 1000
    foi=np.arange(freq[0],freq[1]+1)
    dt = 1/Fs
    wavelet=[]
    tapM = np.zeros((len(foi),N)) 

    for i in range(len(foi)):
        sf = foi[i] / width
        st = 1/(2*np.pi*sf)
        toi2= np.arange(-gwidth*st,gwidth*st, dt)
        A = 1/np.sqrt(st*np.sqrt(np.pi))
        tap = (A*np.exp(-toi2**2/(2*st**2))).astype('csingle')
        acttapnumsmp = len(tap)
        taplen = acttapnumsmp;
        ind  = np.linspace(-(acttapnumsmp-1)/2,(acttapnumsmp-1)/2,acttapnumsmp)*((2*np.pi/Fs)* foi[i])
  ## create wavelet and fft it
##   wav = tap.*exp(i.*ind);
        cosine=tap*np.cos(ind)
        sine= tap*np.sin(ind)
        wav = (cosine+1j*sine).astype('csingle')
        wavelet.append(wav)
        tapm = np.zeros(N)
        tapm[math.ceil(taplen/2)-1:math.ceil(N-taplen/2)-1]=1
        tapM[i,:] = tapm;
    
        
    timeoi = np.array(range(1,N+1))/Fs
   # tapM=tapM[:,::3]
    return wavelet, foi, timeoi,tapM

