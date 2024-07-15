import numpy as np
import matplotlib.pyplot as plt

# Parameters
clock_freq = 10e6  # 10 MHz clock frequency
sampling_rate = 1e9  # 1 GHz sampling rate
duration = 1e-3  # 1 ms duration
jitter_percentage = 0.10  # 10% jitter

# Generate time array
t = np.arange(0, duration, 1/sampling_rate)

# Generate ideal clock signal
ideal_period = 1 / clock_freq
ideal_clock = 0.5 * (1 + np.sign(np.sin(2 * np.pi * clock_freq * t)))

# Generate clock signal with jitter
jitter = np.random.uniform(-jitter_percentage, jitter_percentage, len(t) // int(sampling_rate / clock_freq)) * ideal_period
jittered_t = np.copy(t)
for i in range(1, len(jittered_t) // int(sampling_rate / clock_freq)):
    jittered_t[i * int(sampling_rate / clock_freq): (i + 1) * int(sampling_rate / clock_freq)] += jitter[i]

jittered_clock = 0.5 * (1 + np.sign(np.sin(2 * np.pi * clock_freq * jittered_t)))

# Plot eye diagram function
def plot_eye_diagram(ax, signal, period, title):
    samples_per_period = int(sampling_rate * period)
    num_periods = len(signal) // samples_per_period

    for i in range(num_periods - 1):
        ax.plot(signal[i * samples_per_period:(i + 2) * samples_per_period])
    ax.set_title(title)
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

# Plot FFT function
def plot_fft(ax, signal, sampling_rate, title):
    N = len(signal)
    f = np.fft.fftfreq(N, 1/sampling_rate)
    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal) / N

    ax.plot(f[:N // 2], magnitude[:N // 2])
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot eye diagrams
plot_eye_diagram(axs[0, 0], ideal_clock, ideal_period, 'Eye Diagram of Ideal Clock Signal')
plot_eye_diagram(axs[0, 1], jittered_clock, ideal_period, 'Eye Diagram of Clock Signal with 10% Jitter')

# Plot FFTs
plot_fft(axs[1, 0], ideal_clock, sampling_rate, 'FFT of Ideal Clock Signal')
plot_fft(axs[1, 1], jittered_clock, sampling_rate, 'FFT of Clock Signal with 10% Jitter')

# Adjust layout
plt.tight_layout()
plt.show()
