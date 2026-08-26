import numpy as np
import matplotlib.pyplot as plt


#image A noir de taille 256 x 256 et on ajoute un carre blanc au centre : image ideale avec carre net sans diffusion

N = 256
A = np.zeros((N, N))

A[110:146, 110:146] = 1.0

#image B va etre le noyau ou le kernel de diffusion

x = np.arange(N) - N // 2
y = np.arange(N) - N // 2

X, Y = np.meshgrid(x, y)

sigma = 3 #largeur tahce diffusion donc diff courte ou longue portee

B = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

B = B / B.sum()

#convolution Fourier 

FA = np.fft.fft2(A)

FB = np.fft.fft2(np.fft.ifftshift(B))

C = np.fft.ifft2(FA * FB).real




plt.figure()
plt.imshow(A)
plt.title("A: image ideale")
plt.colorbar()


plt.figure()
plt.imshow(B)
plt.title("B: diffusion kernel")
plt.colorbar()

plt.figure()
plt.imshow(C)
plt.title("C = A convolved with B")
plt.colorbar()



plt.show()


