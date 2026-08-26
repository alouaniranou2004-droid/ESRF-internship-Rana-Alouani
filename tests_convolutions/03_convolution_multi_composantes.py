import numpy as np
import matplotlib.pyplot as plt

N = 256
A = np.zeros((N, N))

A[110:146, 110:146] = 1.0


x = np.arange(N) - N // 2
y = np.arange(N) - N // 2

X, Y = np.meshgrid(x, y)

def gaussian_kernel(X, Y, sigma):
    G =np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    G = G / G.sum()
    return G
    
    
sigma1 = 3
sigma2 = 10
sigma3 = 30

w1 = 0.6
w2 = 0.3
w3 = 0.1


B1 = gaussian_kernel(X, Y, sigma1)
B2 = gaussian_kernel(X, Y, sigma2)
B3 = gaussian_kernel(X, Y, sigma3)

B_total = w1 * B1 + w2 * B2 + w3 * B3
B_total = B_total / B_total.sum()

FA = np.fft.fft2(A)
FB = np.fft.fft2(np.fft.ifftshift(B_total))
C = np.fft.ifft2(FA * FB).real

profile = C[N // 2, :]

plt.figure()
plt.imshow(A)
plt.title("A: image ideale")
plt.colorbar()


plt.figure()
plt.imshow(B1)
plt.title(f"B1: short range component, sigma = {sigma1}, weight ={w1}")
plt.colorbar()

plt.figure()
plt.imshow(B1)
plt.title(f"B2: medium range component, sigma = {sigma2}, weight ={w2}")
plt.colorbar()

plt.figure()
plt.imshow(B1)
plt.title(f"B3: long range component, sigma = {sigma3}, weight ={w3}")
plt.colorbar()


plt.figure()
plt.imshow(B_total)
plt.title(f"B_total: sum of all diffusion components")
plt.colorbar()


plt.figure()
plt.imshow(C)
plt.title("C = A convolved with B_total")
plt.colorbar()

plt.figure()
plt.plot(profile)
plt.title(f"horizontal profile of C with multi component halo")
plt.xlabel("x")
plt.ylabel("intensity") 


center = N // 2

profile_B1 = B1[center, :]
profile_B2 = B2[center, :]
profile_B3 = B3[center, :]
profile_B_total = B_total[center, :]

plt.figure()
plt.plot(profile_B1, label=f"B1 sigma={sigma1}, weight={w1}")
plt.plot(profile_B2, label=f"B2 sigma={sigma2}, weight={w2}")
plt.plot(profile_B3, label=f"B3 sigma={sigma3}, weight={w3}")
plt.plot(profile_B_total, label="B_total", linestyle="--")
plt.title("Horizontal profiles of diffusion kernels")
plt.xlabel("x")
plt.ylabel("kernel value")
plt.legend()





plt.show()


# test git 






















