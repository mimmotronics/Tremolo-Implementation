import numpy as np
import waveforms as waves

'''
Mimmotronics Blog
Tremolo Effect Implementation - January 7th, 2017
Code by Dominic Sciarrino

Implements a tremolo effect on an input signal supplied through the
parameter "n". The sampling frequency of that input signal is fs. Depth and
Rate parameters are passed in through D and fm, respectively. The waveform used
to modulate the input signal can be selected with the waveform parameter.
nargout is used to toggle how many output values to return.
'''

def tremolo(n, fs, D, fm, waveform = 0, nargout = 1):
    t = np.arange(0.0, len(n)*1.0/fs, 1.0/fs)
    if waveform == 1:
        m, n2 = waves.squareWave(D, fm, mode = 0)
    elif waveform == 2:
        m, n2 = waves.sawToothWave(D, fm, mode = 0, durationN = len(n))
    elif waveform == 3:
        m, n2 = waves.triangleWave(D, fm, mode = 0, durationN = len(n))
    else:
        m, n2 = waves.sinusoid(D, fm, durationN = len(n))
    m = m + 1

    result = m * n
    result = result / np.max(result)
    if nargout == 1:
        return result
    else:
        return n2, result
