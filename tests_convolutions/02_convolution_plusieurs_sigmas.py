#test of 2D convolution with several gaussian diffusion kernels


import numpy as np
import matplotlib.pyplot as plt

N = 256
A = np.zeros((N, N))

A[110:146, 110:146] = 1.0


x = np.arange(N) - N // 2
y = np.arange(N) - N // 2

X, Y = np.meshgrid(x, y)

sigmas =[3, 10, 30]

for sigma in sigmas:
    B = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    B = B / B.sum()
    
    FA = np.fft.fft2(A)
    FB = np.fft.fft2(np.fft.ifftshift(B))
    C = np.fft.ifft2(FA * FB).real
    
    profile = C[N // 2, :]
    
    plt.figure()
    plt.imshow(B)
    plt.title(f"B: diffusion kernel, sigma = {sigma}")
    plt.colorbar()
    
    plt.figure()
    plt.imshow(C)
    plt.title(f"C = A convolved with B, sigma = {sigma}")
    plt.colorbar()
    
    
    plt.figure()
    plt.plot(profile)
    plt.title(f"horizontal profile of C, sigma = {sigma}")
    plt.xlabel("x")
    plt.ylabel("intensity")    
    
    
    
plt.figure()
plt.imshow(A)
plt.title("A: image ideale")
plt.colorbar()


plt.show()


