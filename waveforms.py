import numpy as np
import matplotlib as plt

'''
Mimmotronics Blog
Modulation Waveforms - January 7th, 2017
Code by Dominic Sciarrino

Various waveforms used to test modulation based effects.
'''

def squareWave(A, f, durationMS = 500, durationN = 44100, mode = 1, dutyCycle = 0.5, fs = 44100):
    if (mode == 1):
        duration = durationMS / 1000.0
        n_len = duration * fs
        n = np.arange(0.0, n_len, 1.0)
    if (mode == 0):
        n = np.arange(0.0, durationN, 1.0)

    numSampsPerCycle = np.floor(fs/f)
    riseEdgeSamp = np.floor(numSampsPerCycle * (1 - dutyCycle))
    fallEdgeSamp = numSampsPerCycle

    state = 0
    nextRiseEdge = int(riseEdgeSamp)
    nextFallingEdge = int(fallEdgeSamp)
    s = np.array([])
    for i in n:
        currentSample = n[int(i)]
        onRisingEdge = currentSample % nextRiseEdge == 0
        onFallingEdge = currentSample % nextFallingEdge == 0
        if(onRisingEdge == True or currentSample >= nextRiseEdge) and currentSample != 0 and state == 0:
            state = state + 1
            nextRiseEdge = nextRiseEdge + numSampsPerCycle
        if (onFallingEdge == True or currentSample >= nextRiseEdge) and currentSample != 0 and state == 1:
            state = state - 1
            nextFallingEdge = nextFallingEdge + numSampsPerCycle
        if state == 0:
            s = np.append(s, 0)
        if state == 1:
            s = np.append(s, 1)
    s = 2*A * (s - 0.5)
    return s, n

#if mode=1 use durationN as a sample reference
#if mode=0 use durationMS as a time reference
def sinusoid(A, f, phi = 0, durationMS = 500, durationN = 44100, mode = 1, fs = 44100):
    if (mode == 1):
        t = np.arange(0.0, durationN*1.0/fs, 1.0/fs)
    if (mode == 0):
        duration = durationMS / 1000.0
        n_len = duration * fs
        t = np.arange(0.0, n_len/fs, 1.0/fs)

    return A*np.sin(2*3.14159*f*t + phi), t

def sawToothWave(A, f, durationMS = 500, durationN = 44100, mode = 1, fs = 44100):
    if (mode == 1):
        duration = durationMS / 1000.0
        n_len = duration * fs
        n = np.arange(0.0, n_len, 1.0)
    if (mode == 0):
        n = np.arange(0.0, durationN, 1.0)

    numSampsPerCycle = np.floor(fs/f)
    fallEdgeSamp = numSampsPerCycle

    output = 0
    subtractor = 0
    nextFallingEdge = fallEdgeSamp
    s = np.array([])
    for i in n:
        currentSample = n[int(i)]
        onFallingEdge = currentSample % int(nextFallingEdge) == 0
        if (onFallingEdge == True or currentSample >= nextFallingEdge) and currentSample != 0:
            nextFallingEdge = nextFallingEdge + numSampsPerCycle
            output = 0
            subtractor = subtractor + 1
        else:
            output = (int(i) / numSampsPerCycle) - subtractor
        s = np.append(s, output)
    s = 2*A * (s - 0.5)
    return s, n

def triangleWave(A, f, durationMS = 500, durationN = 44100, mode = 1, dutyCycle = 0.5, fs = 44100):
    if (mode == 1):
        duration = durationMS / 1000.0
        n_len = duration * fs
        n = np.arange(0.0, n_len, 1.0)
    if (mode == 0):
        n = np.arange(0.0, durationN, 1.0)

    numSampsPerCycle = np.floor(fs/f)
    turningPointSamp = numSampsPerCycle / 2

    output = -1
    inverter = 1
    nextTurningPoint = turningPointSamp
    s = np.array([])
    for i in n:
        currentSample = n[int(i)]

        onTurningPoint = currentSample % int(nextTurningPoint) == 0
        if (onTurningPoint == True or currentSample >= nextTurningPoint) and currentSample != 0:
            inverter = inverter * (-1)
            nextTurningPoint = nextTurningPoint + turningPointSamp

        output = output + (2/turningPointSamp)*inverter
        s = np.append(s, output)
    s = A * s
    return s, n
