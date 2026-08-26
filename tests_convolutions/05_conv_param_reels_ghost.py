import numpy as np
import matplotlib.pyplot as plt

N = 512
A = np.zeros((N, N))
A[230:282, 230:282] = 1.0


x = np.arange(N) - N // 2
y = np.arange(N) - N // 2

X, Y = np.meshgrid(x, y)


r = np.sqrt(X**2 + Y**2)    #cree distance r de chaqye pixel au centre de l image

scale_l = np.array([115.0, 34.63, 7.414, 3.472, 1.016])
fraction = np.array([0.1859, 0.1590, 0.1196, 0.1375, 0.1971])

replica_factor = 0.007647
replica_shift_x = -47
replica_shift_y = 0

def make_component(r, lamb):
    epsilon = 1e-12

    U = np.exp(-r / lamb) / np.maximum(r, epsilon)

    center = N // 2               #indice du centre de l'image
    U[center, center] = 0.0

    U = U / U.sum()   #chaq pixel de U est divise par la somme tot de tous les pixels de U

    return U

components = []  
    
for lamb in scale_l:
    component = make_component(r, lamb)
    components.append(component)
    
center = N // 2  
    
k0 = np.zeros((N, N))


k0[center, center] = 1.0

for i in range(len(scale_l)):
    k0 = k0 + fraction[i] * components[i]


k0_shifted = np.roll(k0, shift=replica_shift_y, axis=0)
k0_shifted = np.roll(k0_shifted, shift=replica_shift_x, axis=1)

k = k0 + replica_factor * k0_shifted
       


k0 = k0 / k0.sum()


FA = np.fft.fft2(A)
Fk0 = np.fft.fft2(np.fft.ifftshift(k0))
C = np.fft.ifft2(FA * Fk0).real

profile_A = A[center, :]
profile_k0 = k0[center, :]
profile_k = k0[center, :]
profile_C = C[center, :]


plt.figure()
plt.imshow(A)
plt.title("A: ideal image")
plt.colorbar()

plt.figure()
plt.imshow(np.log10(k0 + 1e-20))
plt.title("log10(k0) : main kernel without ghost")
plt.colorbar()


plt.figure()
plt.imshow(np.log10(k + 1e-20))
plt.title("log10(k) : main kernel with ghost")
plt.colorbar()


zoom = 120

plt.figure()
plt.imshow(
    np.log10(k[
      center - zoom:center + zoom,
      center - zoom:center + zoom
    ] + 1e-20)
)
plt.title("Zoom on log10(k): center + possible ghost")
plt.colorbar()

plt.figure()
plt.imshow(C)
plt.title("C = A convolved with K including ghost")
plt.colorbar()


plt.figure()
plt.imshow(
    C[
      center - zoom:center + zoom,
      center - zoom:center + zoom
    ]
)
plt.title("Zoom on C")
plt.colorbar()

plt.figure()
plt.plot(profile_A, label="A: ideal profile")
plt.plot(profile_C, label="C: convolved profile with ghost")
plt.title("Ideal profile vs convolved profile with ghost")
plt.xlabel("x")
plt.ylabel("intensity")
plt.legend()


x_centered = np.arange(N) - center

plt.figure()
plt.semilogy(x_centered, profile_k0 + 1e-20, label="k0 without ghost")
plt.semilogy(x_centered, profile_k + 1e-20, label="k with ghost", linestyle="--")
plt.title("Kernel profile: effect of ghost")
plt.xlabel("x - center")
plt.ylabel("kernel value, log scale")
plt.xlim(-150, 150)
plt.legend()

plt.show()




