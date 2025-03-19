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

mis = -999

###################################################################
# MODULE: kf_filter
# Code written by Jared Rennie based on NCL code from Carl Schreck
###################################################################
def kf_filter(inData,obsPerDay,tMin,tMax,kMin,kMax,hMin,hMax,waveName):
	timeDim = inData.shape[0]
	lonDim = inData.shape[1]
	
	# Reshape data from [time,lon] to [lon,time]
	originalData=np.zeros([lonDim,timeDim],dtype='f')
	for counterX in range(timeDim):
	    test=0
	    for counterY in range(lonDim-1,-1,-1):
	        originalData[test,counterX]=inData[counterX,counterY]
	        test+=1
	
	# Detrend the Data
	detrendData=np.zeros([lonDim,timeDim],dtype='f')
	for counterX in range(lonDim):
	    detrendData[counterX,:]=signal.detrend(originalData[counterX,:])
	
	# Taper 
	taper=signal.windows.tukey(timeDim,0.05,True)
	taperData=np.zeros([lonDim,timeDim],dtype='f')
	for counterX in range(lonDim):
	    taperData[counterX,:]=detrendData[counterX,:]*taper
	
	# Perform 2-D Fourier Transform
	fftData=np.fft.rfft2(taperData)
	kDim=lonDim 
	freqDim=round(fftData.shape[1])
	
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
	    fftData[:, :jMin-1 ] = 0
	if( jMax < ( freqDim - 1 ) ):
	    fftData[:, jMax+1: ] = 0

	if( iMin < iMax ):
	    # Set things outside the range to zero, this is more normal
	    if( iMin > 0 ):
	        fftData[:iMin-1, : ] = 0
	    if( iMax < ( kDim - 1 ) ):
	        fftData[iMax+1:, : ] = 0
	else:
	    # Set things inside the range to zero, this should be somewhat unusual
	    fftData[iMax+1:iMin-1, : ] = 0
	
	# Find constants
	PI = math.pi
	beta = 2.28e-11
	if hMin != -999:
	    cMin = float( 9.8 * float(hMin) )**0.5
	else:
	    cMin=hMin
	if hMax != -999:
	    cMax = float( 9.8 * float(hMax) )**0.5
	else:
	    cMax=hMax
	c = np.array([cMin,cMax])
	spc = 24 * 3600. / ( 2 * PI * obsPerDay ) # seconds per cycle
	
	# Now set things to zero that are outside the Kelvin dispersion
	for i in range(0,kDim):
		# Find Non-Dimensional WaveNumber (k)
		if( i > ( kDim / 2 ) ):
			# k is negative
			k = ( i - kDim  ) * 1. / (6.37e6) # adjusting for circumfrence of earth
		else:
			# k is positive
			k = i * 1. / (6.37e6) # adjusting for circumfrence of earth
		
		# Find Frequency
		freq = np.array([ 0, freqDim * (1. / spc) ]) #waveName='None'
		jMinWave = 0
		jMaxWave = freqDim	
		if waveName.lower() == "kelvin":
		    freq = k * c
		if waveName.lower() == "er":
		    freq = -beta * k / ( k**2 + 3. * beta / c )
		if waveName.lower() == "ig1":
		    freq = ( 3 * beta * c + k**2 * c**2 )**0.5
		if waveName.lower() == "ig2":
		    freq = ( 5 * beta * c + k**2 * c**2 )**0.5
		if waveName.lower() == "mrg" or waveName.lower()=="ig0":   	
			if( k == 0 ):
				freq = ( beta * c )**0.5
			else:
				if( k > 0):
					freq = k * c * ( 0.5 + 0.5 * ( 1 + 4 * beta / ( k**2 * c ) )**0.5 )
				else:
					freq = k * c * ( 0.5 - 0.5 * ( 1 + 4 * beta / ( k**2 * c ) )**0.5 )	
		
		# Get Min/Max Wave 
		if(hMin==mis):
		    jMinWave = 0
		else:
		    jMinWave = int( math.floor( freq[0] * spc * timeDim ) )

		if(hMax==mis):
		    jMaxWave = freqDim
		else:
		    jMaxWave = int( math.ceil( freq[1] * spc * timeDim ) )

		jMaxWave = max(jMaxWave, 0)
		jMinWave = min(jMinWave, freqDim)
		
		# set the appropriate coefficients to zero
		i=int(i)
		jMinWave=int(jMinWave)
		jMaxWave=int(jMaxWave)
		if( jMinWave > 0 ):
		    fftData[i, :jMinWave-1] = 0
		if( jMaxWave < ( freqDim - 1 ) ):
		    fftData[i, jMaxWave+1:] = 0
	
	# perform the inverse transform to reconstruct the data
	returnedData=np.fft.irfft2(fftData) 
	
	# Reshape data from [lon,time] to [time,lon]
	outData=np.zeros([timeDim,lonDim],dtype='f')
	for counterX in range(returnedData.shape[1]):
	    test=0
	    for counterY in range(lonDim-1,-1,-1):
	        outData[counterX,counterY]=returnedData[test,counterX] 
	        test+=1
    
    # Return Result
	return outData

