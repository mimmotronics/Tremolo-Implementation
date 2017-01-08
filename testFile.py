import utilities as util
import effectBank as effects

'''
Mimmotronics Blog
Testbench file for effectBank.py. Utilizes the utilities library from
sms-tools (search for it) in order to read and write WAV files.
'''

fs, x = util.wavread("Example1.wav")
ftrem = 9
D = 0.97
n, out_d1 = effects.tremolo(x, 44100, D, ftrem, 1, 0)
util.wavwrite(out_d1, 44100.0, 'out_d1.wav')
