"""Synthesize the intro soundtrack: QRS beeps synced to the ECG, medium-priority alarm, trust tone, room tone."""
import numpy as np, wave
SR, DUR = 48000, 7.5
n = int(SR * DUR); t = np.arange(n) / SR
out = np.zeros(n)
rng = np.random.default_rng(0)
out += np.convolve(rng.normal(0, 1, n), np.ones(400) / 400, mode="same") * 0.02     # room tone


def tone(start, dur, f, amp, attack=0.004, release=0.05):
    i0 = int(start * SR); m = int(dur * SR); tt = np.arange(m) / SR
    env = np.minimum(1, tt / attack) * np.exp(-np.maximum(0, tt - (dur - release)) / (release / 3))
    sig = amp * env * (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt))
    out[i0:i0 + m] += sig[: max(0, min(m, n - i0))]


HR, k = 86, 0
while (tb := (k + 0.30) * 60 / HR) < DUR - 0.2:          # R-peak at phase 0.30 of each cycle
    if tb > 0.35:
        tone(tb, 0.07, 1020, 0.10)
    k += 1
T_ALERT, T_TRUST = 3.78, 4.8                             # risk crosses 3.0% / LOW TRUST banner
for i in range(3):
    tone(T_ALERT + i * 0.2, 0.15, 659, 0.30, release=0.03)   # IEC 60601-1-8 style medium-priority burst
tone(T_TRUST, 0.22, 880, 0.22, release=0.12); tone(T_TRUST + 0.24, 0.32, 587, 0.22, release=0.2)
out *= np.minimum(1, t / 0.5) * np.clip((DUR - t) / 0.6, 0, 1)
out = np.clip(out, -1, 1)
with wave.open("intro_audio.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out * 32767 * 0.9).astype(np.int16).tobytes())
