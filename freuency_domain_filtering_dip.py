""" Frequency Domain Adaptive Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

def fdaf(x, d, M, mu=0.05, beta=0.9):
  H = np.zeros(M+1,dtype=np.complex)
  norm = np.full(M+1,1e-8)

  window =  np.hanning(M)
  x_old = np.zeros(M)

  num_block = min(len(x),len(d)) // M
  e = np.zeros(num_block*M)

  for n in range(num_block):
    x_n = np.concatenate([x_old,x[n*M:(n+1)*M]])
    d_n = d[n*M:(n+1)*M]
    x_old = x[n*M:(n+1)*M]

    X_n = fft(x_n)
    y_n = ifft(H*X_n)[M:]
    e_n = d_n-y_n
    e[n*M:(n+1)*M] = e_n

    e_fft = np.concatenate([np.zeros(M),e_n*window])
    E_n = fft(e_fft)

    norm = beta*norm + (1-beta)*np.abs(X_n)**2
    G = mu*E_n/(norm+1e-3)
    H = H + X_n.conj()*G

    h = ifft(H)
    h[M:] = 0
    H = fft(h)

  return e