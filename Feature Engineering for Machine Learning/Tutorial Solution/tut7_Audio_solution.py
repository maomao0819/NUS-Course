from matplotlib import pyplot as plt
import numpy as np
import sounddevice as sd


# https://www.szynalski.com/tone-generator/

if (0):
    # Question 1
    #----- record audio
    fs       = 44100  # Sample frequency
    duration = 5      # duration in seconds

    myRecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('tone10000Hz.wav', fs, myRecording)  # save into a .wav file

if (0):
    # Question 2
    #----- read audio signal from .wav file
    import soundfile as sf
    #y, fs = sf.read('myRecording.wav')
    y, fs = sf.read('tone1000Hz.wav')
    #y, fs = sf.read('tone2000Hz.wav')
    #y, fs = sf.read('tone5000Hz.wav')
    #y, fs = sf.read('tone10000Hz.wav')
    sd.play(y, fs)
    sd.wait()

    # Question 3
    #----- FFT on the signal
    from scipy.fft import fft, fftfreq

    sampling_interval = 1/fs

    print('sampling frequency=%f,  sampling interval=%f ' % (fs, sampling_interval))

    #N = np.int32(fs * duration)
    N = len(y)
    print('number of points N = %d ' % N)

    Y = fft(y, N)
    freq_step = fftfreq(N, sampling_interval)

    # --- frequency resolution
    fres = fs / N
    print('frequency resolution is %f' % fres)

    print('freq_step[0],[1],[2]=%f, %f, %f ' % (freq_step[0], freq_step[1], freq_step[2]))

    print('freq_step[-2]=%f' % freq_step[-2])

    #--- find where the peaks are
    peaks = np.where(np.abs(Y) > 2000)
    print('location of peaks :', peaks)

    #print(np.abs(Y[800:1150]))
    #plt.plot(freq_step, np.abs(Y))
    plt.plot(np.abs(Y))
    plt.show()

