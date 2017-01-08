# -*- coding: utf-8 -*-
"""
General Utilities

Functions I've gathered over the development of my personal audio processing toolbox.
Many of these are from Stanford University's Coursera MOOC on "Audio Signal Processing 
for Music Applications". Refer to that course for more great material on Audio Processing 
with Python.
"""

import os, sys, copy
from scipy.io.wavfile import write, read
from scipy.fftpack import fft, ifft
import numpy as np
import winsound, subprocess
import math

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

winsound_imported = False	
if sys.platform == "win32":
	try:
		import winsound
		winsound_imported = True
	except:
		print "You won't be able to play sounds, winsound could not be imported"

def isPower2(N):
    return ((N & (N-1)) == 0) and N > 0
    
def wavread(filename):
	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""

	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		raise ValueError("Input file is wrong")

	fs, x = read(filename)

	if (len(x.shape) !=1):                                   # raise error if more than one channel
		raise ValueError("Audio file should be mono")

	if (fs !=44100):                                         # raise error if more than one channel
		raise ValueError("Sampling rate of input sound should be 44100")

	#scale down and convert audio into floating point number in range of -1 to 1
	x = np.float32(x)/norm_fact['float32']#x.dtype.name]
	return fs, x

def wavplay(filename):
	"""
	Play a wav audio file from system using OS calls
	filename: name of file to read
	"""
	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		print("Input file does not exist. Make sure you computed the analysis/synthesis")
	else:
		if sys.platform == "linux" or sys.platform == "linux2":
		    # linux
		    subprocess.call(["aplay", filename])

		elif sys.platform == "darwin":
			# OS X
			subprocess.call(["afplay", filename])
		elif sys.platform == "win32":
			if winsound_imported:
				winsound.PlaySound(filename, winsound.SND_FILENAME)
			else:
				print("Cannot play sound, winsound could not be imported")
		else:
			print("Platform not recognized")
   
def wavwrite(y, fs, filename):
	"""
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""

	x = copy.deepcopy(y)                         # copy array
	x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
	x = np.int16(x)                              # converting to int16 type
	write(filename, fs, x)
 
def impulseExtractor(x_clean, x_effect):
    fs, x_clean1 = wavread(x_clean)
    fs, x_effect1 = wavread(x_effect)
    
    N = len(x_clean1)
    X_clean = np.fft.fft(x_clean1)
    X_effect = np.fft.fft(x_effect1)
    
    H = X_effect / X_clean
    
    return fs, N, H

    
def dftAnal(x, w, N):
    """
    INPUTS: signal:    x
             window:    w
             dft size   N
             
     The signal x and the window w should be the same size. This is determined by 
     the window size M. The window is created using the lines:
     
         from scipy.signal import get_window
         w = get_window('hamming',M)
     
     where 'hamming' can be any valid window that scipy supports. The signal to
     be analyzed by this function needs to be the same size M as the window above
     so we have to cut a snippet of sound from the main signal. We do this in 
     terms of the time at which we wish to start the cut:
     
         time = 0.5
         x1 = x[int(time*fs):int(time*fs)+M]
     
     The operation time*fs gives the sample point at which to start the cut. 
     The type-cast to int ensures an integer is being supplied to the command.
     
     _______
     
     OUTPUTS: magnitude     mX
              phase         pX
              window size   M
              
     The magintude is in dB scale. The phase spectrum is phase-unwrapped, and 
     the window size is the length of the output spectrum. Both mX and pX only 
     show the positive frequency values because of the symmetry real signals 
     have in Fourier Transformations.
    """
 
    tol = 1e-14
    if not(isPower2(N)):                                 # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if (w.size > N):                                        # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N/2)+1                                            # size of positive spectrum, it includes sample 0
    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x*w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]        
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values         
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX

def dftSynth(mX, pX, M):
    """
    INPUTS: mX     Magnitude of X
             pX     Phase of X
             M      Window Size
             
     The magnitude and phase spectrums are only values corresponding to 
     the positive frequency axis.
     
     ______
     
     OUTPUTS: y     Synthesized Output Signal
     
     The synthesized signal is a real-values signal corresponding to the 
     windowed section of the original input signal x.
     
     NOTE: The signal y MUST BE NORMALIZED with respect to the window function

     dftSynth(mX, pX, M) * sum(w)

     Where w is the window function.      
     
    """

    hN = mX.size                                            # size of positive spectrum, it includes sample 0
    N = (hN-1)*2                                            # FFT size
    if not(isPower2(N)):                                 # raise error if N not a power of two, thus mX is wrong
        raise ValueError("size of mX is not (N/2)+1")

    hM1 = int(math.floor((M+1)/2))                          # half analysis window size by rounding
    hM2 = int(math.floor(M/2))                              # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    y = np.zeros(M)                                         # initialize output array
    Y = np.zeros(N, dtype = complex)                        # clean output spectrum
    Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                    # generate positive frequencies
    Y[hN:] = 10**(mX[-2:0:-1]/20) * np.exp(-1j*pX[-2:0:-1]) # generate negative frequencies
    fftbuffer = np.real(ifft(Y))                            # compute inverse FFT
    y[:hM2] = fftbuffer[-hM2:]                              # undo zero-phase window
    y[hM2:] = fftbuffer[:hM1]
    return y

def stftAnal(x, fs, w, N, H) :
	"""
	Analysis of a sound using the short-time Fourier transform
	x: input array sound, w: analysis window, N: FFT size, H: hop size
	returns xmX, xpX: magnitude and phase spectra
	"""
	if (H <= 0):                                   # raise error if hop size 0 or negative
		raise ValueError("Hop size (H) smaller or equal to 0")
		
	M = w.size                                      # size of analysis window
	hM1 = int(math.floor((M+1)/2))                  # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                      # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                  # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                  # add zeros at the end to analyze last sample
	pin = hM1                                       # initialize sound pointer in middle of analysis window       
	pend = x.size-hM1                               # last sample to start a frame
	w = w / sum(w)                                  # normalize analysis window
	while pin<=pend:                                # while sound pointer is smaller than last sample      
		x1 = x[pin-hM1:pin+hM2]                       # select one frame of input sound
		mX, pX = dftAnal(x1, w, N)                # compute dft
		if pin == hM1:                                # if first frame create output arrays
			xmX = np.array([mX])
			xpX = np.array([pX])
		else:                                         # append output to existing array 
			xmX = np.vstack((xmX,np.array([mX])))
			xpX = np.vstack((xpX,np.array([pX])))
		pin += H                                      # advance sound pointer
	return xmX, xpX


def stftSynth(mY, pY, M, H) :
	"""
	Synthesis of a sound using the short-time Fourier transform
	mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
	returns y: output sound
	"""
	hM1 = int(math.floor((M+1)/2))                   # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                       # half analysis window size by floor
	nFrames = mY[:,0].size                           # number of frames
	y = np.zeros(nFrames*H + hM1 + hM2)              # initialize output array
	pin = hM1                  
	for i in range(nFrames):                         # iterate over all frames      
		y1 = dftSynth(mY[i,:], pY[i,:], M)         # compute idft
		y[pin-hM1:pin+hM2] += H*y1                     # overlap-add to generate output sound
		pin += H                                       # advance sound pointer
	y = np.delete(y, range(hM2))                     # delete half of first window which was added in stftAnal
	y = np.delete(y, range(y.size-hM1, y.size))      # delete the end of the sound that was added in stftAnal
	return y

def peakDetection(mX, t):
	"""
	Detect spectral peak locations
	mX: magnitude spectrum, t: threshold
	returns ploc: peak locations
	"""

	thresh = np.where(mX[1:-1]>t, mX[1:-1], 0);             # locations above threshold
	next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
	prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
	ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
	ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
	return ploc

def peakInterp(mX, pX, ploc):
	"""
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

	val = mX[ploc]                                          # magnitude of peak bin
	lval = mX[ploc-1]                                       # magnitude of bin at left
	rval = mX[ploc+1]                                       # magnitude of bin at right
	iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
	ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
	ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
	return iploc, ipmag, ipphase