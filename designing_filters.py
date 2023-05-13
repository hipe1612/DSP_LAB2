import matplotlib.pyplot as plt
import scipy.fftpack as fft
import scipy.signal as sgl
import numpy as np

Input_1kHz_15kHz =[

+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
-0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
+0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
-0.8660254038, -0.4619397663, -1.3194792169, -1.1827865776, -0.5000000000, -1.1827865776, -1.3194792169, -0.4619397663, 
-0.8660254038, -1.2552931065, -0.3535533906, -0.4174197128, -1.0000000000, -0.1913417162, +0.0947343455, -0.5924659585, 
-0.0000000000, +0.5924659585, -0.0947343455, +0.1913417162, +1.0000000000, +0.4174197128, +0.3535533906, +1.2552931065, 
+0.8660254038, +0.4619397663, +1.3194792169, +1.1827865776, +0.5000000000, +1.1827865776, +1.3194792169, +0.4619397663, 
+0.8660254038, +1.2552931065, +0.3535533906, +0.4174197128, +1.0000000000, +0.1913417162, -0.0947343455, +0.5924659585, 
+0.0000000000, -0.5924659585, +0.0947343455, -0.1913417162, -1.0000000000, -0.4174197128, -0.3535533906, -1.2552931065, 
]


# FIR filters
def fir():
    while True:
        print("1 - Bartlett window")
        print("2 - Hamming window")
        print("3 - Blackman window")
        print("4 - exit")

        choice = int(input("choice: "))

        match choice:
            case 1:
                bartlett()
            case 2:
                hamming()
            case 3:
                blackman()
            case 4:
                break
            case _:
                print("invalid choice!")

def bartlett():
    window = sgl.windows.bartlett(51)

    plt.subplot(2, 1, 1)
    plt.plot(window)
    plt.xlabel("Time representation of Bartlett window")

    norm = fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(norm))
    response = np.abs(fft.fftshift(norm / abs(norm).max()))

    plt.subplot(2, 1, 2)
    plt.plot(freq, response)
    # plt.plot(fft.fft(window), markerfmt=" ")
    # plt.axis([-0.5, 0.5, -120, 0])
    plt.xlabel("Frequency representation of Bartlett window")
    plt.show()

    output = np.convolve(Input_1kHz_15kHz, window)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    freq_input = np.abs(fft.fft(Input_1kHz_15kHz))
    plt.subplot(2, 2, 2)
    plt.plot(freq_input)
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()


def hamming():
    window = sgl.windows.hamming(51)

    plt.subplot(2, 1, 1)
    plt.plot(window)
    plt.xlabel("Time representation of Hamming window")

    norm = fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(norm))
    response = np.abs(fft.fftshift(norm / abs(norm).max()))

    plt.subplot(2, 1, 2)
    plt.plot(freq, response)
    # plt.plot(fft.fft(window), markerfmt=" ")
    # plt.axis([-0.5, 0.5, -120, 0])
    plt.xlabel("Frequency representation of Hamming window")
    plt.show()

    output = np.convolve(Input_1kHz_15kHz, window)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    freq_input = np.abs(fft.fft(Input_1kHz_15kHz))
    plt.subplot(2, 2, 2)
    plt.plot(freq_input)
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()


def blackman():
    window = sgl.windows.blackman(51)

    plt.subplot(2, 1, 1)
    plt.plot(window)
    plt.xlabel("Time representation of Blackman window")

    norm = fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(norm))
    response = np.abs(fft.fftshift(norm / abs(norm).max()))

    plt.subplot(2, 1, 2)
    plt.plot(freq, response)
    # plt.plot(fft.fft(window), markerfmt=" ")
    # plt.axis([-0.5, 0.5, -120, 0])
    plt.xlabel("Frequency representation of Blackman window")
    plt.show()

    output = np.convolve(Input_1kHz_15kHz, window)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    freq_input = np.abs(fft.fft(Input_1kHz_15kHz))
    plt.subplot(2, 2, 2)
    plt.plot(freq_input)
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()

def iir():
    while True:
        print("1 - Chebyshev filter")
        print("2 - Butterworth filter")
        print("3 - Bessel filter")
        print("4 - exit")

        choice = int(input("choice: "))

        match choice:
            case 1:
                chebyshev()
            case 2:
                butterworth()
            case 3:
                bessel()
            case 4:
                break
            case _:
                print("invalid choice!")


def chebyshev():
    b, a = sgl.cheby1(10, 1, 15, "low", fs=len(Input_1kHz_15kHz))
    w, h = sgl.freqs(b, a)

    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.xlabel("Frequency representation of Chebyshev filter")
    plt.show()

    output = sgl.filtfilt(b, a, Input_1kHz_15kHz)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(fft.fft(Input_1kHz_15kHz)))
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()


def butterworth():
    b, a = sgl.butter(10, 15, "low", fs=len(Input_1kHz_15kHz))
    w, h = sgl.freqs(b, a)

    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.xlabel("Frequency representation of Butterworth filter")
    plt.show()

    output = sgl.filtfilt(b, a, Input_1kHz_15kHz)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(fft.fft(Input_1kHz_15kHz)))
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()


def bessel():
    b, a = sgl.bessel(10, 15, "low", fs=len(Input_1kHz_15kHz))
    w, h = sgl.freqs(b, a)

    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.xlabel("Frequency representation of Bessel filter")
    plt.show()

    output = sgl.filtfilt(b, a, Input_1kHz_15kHz)

    plt.subplot(2, 2, 1)
    plt.plot(Input_1kHz_15kHz, "r.", ms=2)
    plt.xlabel("Input in time domain")

    plt.subplot(2, 2, 2)
    plt.plot(np.abs(fft.fft(Input_1kHz_15kHz)))
    plt.xlabel("Input in frequency domain")

    plt.subplot(2, 2, 3)
    plt.plot(output, "r.", ms=2)
    plt.xlabel("Output in time domain")

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(fft.fft(output)))
    plt.xlabel("Output in frequency domain")

    plt.show()

def designing_filters_init():
    while True:
        print("1 - FIR")
        print("2 - IIR")
        print("3 - exit")

        choice = int(input("choice: "))

        match choice:
            case 1:
                fir()
            case 2:
                iir()
            case 3:
                break
            case _:
                print("invalid choice!")

if __name__ == "__main__":
    designing_filters_init()