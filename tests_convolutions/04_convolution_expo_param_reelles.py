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


def make_component(r, lamb):
    epsilon = 1e-12

    U = np.exp(-r / lamb) / np.maximum(r, epsilon)

    center = N // 2               #indice du centre de l'image
    U[center, center] = 0.0

    U = U / U.sum()   #chaq pixel de U est divise par la somme tot de tous les pixels de U

    return U
    
components =[]

for lamb in scale_l:
    component = make_component(r, lamb)
    components.append(component)  
    
    
    
    
center = N // 2  
    
K0 = np.zeros((N, N))


K0[center, center] = 1.0

for i in range(len(scale_l)):
    K0 = K0 + fraction[i] * components[i]
    
K0 = K0 / K0.sum()

FA = np.fft.fft2(A)
FK0 = np.fft.fft2(np.fft.ifftshift(K0))
C = np.fft.ifft2(FA * FK0).real

profile_A = A[center, :]
profile_K0 = K0[center, :]
profile_C = C[center, :]



plt.figure()
plt.imshow(A)
plt.title("A: ideal image")
plt.colorbar()

plt.figure()
plt.imshow(np.log10(K0 + 1e-20))
plt.title("K0: delta + exponential diffusion components = total kernel in log scale")
plt.colorbar()

plt.figure()
plt.imshow(C)
plt.title("C = A convolved with K0")
plt.colorbar()

plt.figure()
plt.plot(profile_A, label="A: ideal profile")
plt.plot(profile_C, label="C: convolved profile")
plt.title("Ideal profile vs convolved profile")
plt.xlabel("x")
plt.ylabel("intensity")
plt.legend()

plt.figure()

for i in range(len(scale_l)):
    profile_component = components[i][center, :]
    plt.semilogy(
        profile_component + 1e-20,
        label=f"scale_l={scale_l[i]}, fraction={fraction[i]}"
    )
    

plt.semilogy(profile_K0 + 1e-20, label="K0 total", linestyle="--")
plt.title("Profiles of exponential components and K0 - log scale")
plt.xlabel("x")
plt.ylabel("kernel value")
plt.legend()

plt.show()








