# Import modules
import numpy as np
import numpy.ma as ma
import os, sys, time, re
from datetime import date
import scipy
import calendar
import math
##import netCDF4 as nc
from scipy import signal,fftpack, interpolate
from scipy.interpolate import griddata
#added by RRB
import xarray as xr
# for debugging
import matplotlib.pyplot as plt
from IPython.display import display

mis = -999

###################################################################
# MODULE: kf_filter_plus_PCF_projection
# Code written by Rosimar Rios-Berrios, adapted from code 
# originally written by Jared Rennie based on NCL code from Carl Schreck
###################################################################
def timeSpaceFilter(inData,obsPerDay,tMin,tMax,kMin,kMax):
    #inData format: [lat,lon,time]
    timeDim = inData.shape[2]
    lonDim = inData.shape[1]
    latDim = inData.shape[0]
        
    # Reshape data from [time,lon] to [lon,time]
    #originalData=np.zeros([lonDim,timeDim],dtype='f')
    #for counterX in range(timeDim):
    #    test=0
    #    for counterY in range(lonDim-1,-1,-1):
    #        originalData[test,counterX]=inData[counterX,counterY]
    #        test+=1
    # RRB: reverse the longitude axis. This seems necessary to distinguish between east/westward wavenumbers.
    originalData=inData[:,::-1,:]

    # Detrend the Data
    detrendData=np.zeros([latDim,lonDim,timeDim],dtype='f')
    for counterX in range(lonDim):
        detrendData[:,counterX,:]=signal.detrend(originalData[:,counterX,:])
    detrendData=originalData    
    
    # Taper 
    taper=signal.tukey(timeDim,0.05,True)
    taperData=np.zeros([latDim,lonDim,timeDim],dtype='f')
    for counterX in range(lonDim):
        taperData[:,counterX,:]=detrendData[:,counterX,:]*taper
    taperData=originalData
    
    # Perform 2-D Fourier Transform
    fftData=np.fft.rfft2(taperData)
    kDim=lonDim 
    freqDim=round(fftData.shape[2])
    
    # Find the indeces for the period cut-offs
    jMin = int(round( ( timeDim * 1. / ( tMax * obsPerDay ) ), 0 ))
    jMax = int(round( ( timeDim * 1. / ( tMin * obsPerDay ) ), 0 ))
    jMax = min( ( jMax, freqDim ) )

    # Find the indices for the wavenumber cut-offs
    # This is more complicated because east and west are separate
    if( kMin < 0 ):
        iMin = round( ( kDim + kMin ), 3 )
        iMin = max( ( iMin, ( kDim / 2 ) ) )
    else:
        iMin = round( kMin, 3 )
        iMin = min( ( iMin, ( kDim / 2 ) ) )

    if( kMax < 0 ):
        iMax = round( ( kDim + kMax ), 3 )
        iMax = max( ( iMax, ( kDim / 2 ) ) )
    else:
        iMax = round( kMax, 3 )
        iMax = min( ( iMax, ( kDim / 2 ) ) )
      
    # set the appropriate coefficients to zero
    iMin=int(iMin)
    iMax=int(iMax)
    jMin=int(jMin)
    jMax=int(jMax)
    if( jMin > 0 ):
        fftData[:, :, :jMin-1 ] = 0
    if( jMax < ( freqDim - 1 ) ):
        fftData[:, :, jMax+1: ] = 0

    if( iMin < iMax ):
        # Set things outside the range to zero, this is more normal
        if( iMin > 0 ):
            fftData[:, :iMin-1, : ] = 0
        if( iMax < ( kDim - 1 ) ):
            fftData[:, iMax+1:, : ] = 0
    else:
        # Set things inside the range to zero, this should be somewhat unusual
        fftData[:, iMax+1:iMin-1, : ] = 0
    
    
    # Return FFT coefficients -- these are not the final data!
    return fftData

# define functions for PCF
def pcfDn(lat,lat_0,deg):
    
    arg = ( lat /( ( 2**0.5 )*lat_0 ) )
    if (deg == 0):
        Dn = np.exp(-0.25*((lat/lat_0)**2.0))
    elif (deg == 1):
        Dn = np.exp(-0.25*((lat/lat_0)**2.0)) * ( arg )
    elif (deg == 2):
        Dn = np.exp(-0.25*((lat/lat_0)**2.0)) * ((arg**2.0) - 1.0)
        
    Dn = xr.DataArray(Dn,dims=['lat'],coords=[lat])
    
    return Dn

def kf_filter_plus_PCF_projection(u,v,Z,lat,obsPerDay,tMin,tMax,kMin,kMax,waveName):
    
    # define constants after Yang et al. 2003, 2021
    print('WARNING: temporarily changed the trapping y to 3 degrees, instead of 6')
    lat_0 = 6. #6 degrees
    g = 9.81
    Ce = 20.0 
    
    # calculate functions for FFT analysis 
    q = u + (g/Ce)*Z
    r = u - (g/Ce)*Z
    
    # get FFT coefficients for desired wavenumber and time periods
    qfftData = timeSpaceFilter(q,obsPerDay,tMin,tMax,kMin,kMax)
    # convert into xarray data arrays for manipulation later on
    qfftData = xr.DataArray(qfftData,dims=['lat','k','f'],\
                            coords=[lat,np.arange(0,qfftData.shape[1]),np.arange(0,qfftData.shape[2])])

    if (waveName == 'Kelvin'):
        # the solution is q0D0
        D0 = pcfDn(lat,lat_0,0)
        # project q onto D0
        qfftData = qfftData*D0
        # we can now filter back
        qDataReturned=np.fft.irfft2(qfftData)        
        # do nothing to r or v
        rDataReturned=np.zeros(qDataReturned.shape)
        vDataReturned=np.zeros(qDataReturned.shape)
    else:
        # if wave name is other than Kelvin, we also need to filter v and r
        vfftData = timeSpaceFilter(v,obsPerDay,tMin,tMax,kMin,kMax)
        vfftData = xr.DataArray(vfftData,dims=['lat','k','f'],\
                            coords=[lat,np.arange(0,vfftData.shape[1]),np.arange(0,vfftData.shape[2])])

        # projections go here
        if (waveName == 'WMRG'):
            #solution is q1D1,v0D0, and r = 0
            rDataReturned=np.zeros(r.shape)
            D0 = pcfDn(lat,lat_0,0)
            D1 = pcfDn(lat,lat_0,1)
            qfftData = qfftData*D1
            vfftData = vfftData*D0
        elif (waveName == 'ER1') | (waveName == 'IG1'):
            rfftData = timeSpaceFilter(r,obsPerDay,tMin,tMax,kMin,kMax)
            rfftData = xr.DataArray(rfftData,dims=['lat','k','f'],\
                                coords=[lat,np.arange(0,rfftData.shape[1]),np.arange(0,rfftData.shape[2])])
            #solution is q2D2, v1D1, & r0D0
            D0 = pcfDn(lat,lat_0,0)
            D1 = pcfDn(lat,lat_0,1)            
            D2 = pcfDn(lat,lat_0,2)
            qfftData = qfftData*D2
            vfftData = vfftData*D1
            rfftData = rfftData*D0
            
            # we can now filter back
            rDataReturned=np.fft.irfft2(rfftData)
        
        print(rDataReturned.min(),rDataReturned.max())
        #filter back the rest of variables
        qDataReturned=np.fft.irfft2(qfftData)
        print(qDataReturned.min(),qDataReturned.max())
        vDataReturned=np.fft.irfft2(vfftData)
        print(vDataReturned.min(),vDataReturned.max())

        
    #recover u and Z
    # u = (q+r)/2; z = (Ce/g)*(q-r)/2
    uDataReturned = (rDataReturned+qDataReturned)/2.0
    ZDataReturned = (Ce/g)*(qDataReturned-rDataReturned)/2.0
    
    # reverse back the longitude axis & convert into xarray data
    uDataOut=uDataReturned[:,::-1,:]#,dims=u.dims,coords=u.coords)
    vDataOut=vDataReturned[:,::-1,:]#,dims=v.dims,coords=v.coords)
    ZDataOut=ZDataReturned[:,::-1,:]#,dims=Z.dims,coords=Z.coords)
    
    ## Reshape data from [lon,time] to [time,lon]
    #outData=np.zeros([timeDim,lonDim],dtype='f')
    #for counterX in range(returnedData.shape[1]):
    #    test=0
    #    for counterY in range(lonDim-1,-1,-1):
    #        outData[counterX,counterY]=returnedData[test,counterX] 
    #        test+=1
    
    # Return Result
    return uDataOut, vDataOut, ZDataOut
