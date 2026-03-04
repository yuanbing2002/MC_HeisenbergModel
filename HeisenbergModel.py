import numpy as np
import matplotlib.pyplot as plt

# adjustable parameters
N = 5 # the number of unit cells in a period
loop = 200  # loop times for Metropolis Algorithm to obtain Markov process
time = 10  # Metropolis times for a specific temperature
Tstep = 1  # step for temperature
Tinterval = [1e-4,10.]

# constants
global kB,J
kB = 1  # Boltzmann constant
J = 1

# generate a vector uniformly distributed on unit sphere
def Sgenerator():
    theta = 2*np.pi*np.random.rand()
    phi = np.arccos(1-2*np.random.rand())
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return [x,y,z]

# input the position and the state S, return default energy and energy of the new spin
def getEnergy(mode,i,j,k,S,newspin):
    i_prime = i + 1 if i < N - 1 else 0
    j_prime = j + 1 if j < N - 1 else 0
    k_prime = k + 1 if k < N - 1 else 0
    if mode == 0:  # calculate the energy of S[1,i,j,k](cubic lattice)
        sumspin = S[1,i,j,k] + S[1,i,j-1,k] + S[1,i-1,j-1,k] + S[1,i-1,j,k]+S[2,i,j,k]+S[2,i,j,k-1]+S[2,i,j-1,k-1]+S[2,i,j-1,k]+S[3,i,j,k]+S[3,i,j,k-1]+S[3,i-1,j,k-1]+S[3,i-1,j,k]
        s0 = S[0,i,j,k]
    elif mode == 1:  # calculate the energy of S[2,i,j,k](up-down face-centered)
        s0 = S[1,i,j,k]
        sumspin = S[0,i,j,k]+S[0,i_prime,j,k] + S[0,i_prime,j_prime,k] + S[0,i,j_prime,k]+S[2,i,j,k] + S[2,i_prime,j,k] + S[2,i_prime,j,k-1] + S[2,i,j,k-1]+S[3,i,j,k] + S[3,i,j_prime,k] + S[3,i,j_prime,k-1] + S[3,i,j,k-1]
    elif mode == 2:   # calculate the energy of S[3,i,j,k](left-right face-centered)
        s0 = S[2,i,j,k]
        sumspin = S[3,i,j,k] + S[3,i,j_prime,k] + S[3,i-1,j_prime,k] + S[3,i-1,j,k]+S[1,i,j,k] + S[1,i,j,k_prime] + S[1,i-1,j,k_prime]+ S[1,i-1,j,k]+S[0,i,j,k] + S[0,i,j_prime,k] + S[0,i,j_prime,k_prime] + S[0,i,j,k_prime]
    elif mode == 3:   # calculate the energy of S[4,i,j,k](front-back face-centered)
        s0 = S[3,i,j,k]
        sumspin = S[2,i,j,k] + S[2,i_prime,j,k] + S[2,i_prime,j-1,k] + S[2,i,j-1,k]+S[0,i,j,k] + S[0,i_prime,j,k] + S[0,i_prime,j,k_prime] + S[0,i,j,k_prime]+S[1,i,j,k] + S[1,i,j,k_prime] + S[1,i,j-1,k_prime] + S[1,i,j-1,k]
    H1 = -J * np.dot(s0, sumspin)  # original spin energy
    H2 = -J * np.dot(newspin, sumspin)  # new spin energy
    return H1, H2

# input state S and temperature T, return new state(exchange all spin once)
def Metropolis(S,T):
    for mode in range(4):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    newspin = Sgenerator()
                    energyBefore,energyLater = getEnergy(mode,i, j, k, S,newspin)
                    alpha = min(1.0, np.exp(-(energyLater - energyBefore) / (kB * T)))
                    if np.random.rand() <= alpha:
                        S[mode,i, j, k] = newspin
                    else:
                        pass
    return S

# for a specific temperature, get the equilibrium
def getEquilibrium(N,temperature,loop):
    # initial state
    S = np.zeros((4,N, N, N, 3),dtype=float)
    for n in range(4):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    S[n,i,j,k] = [0.,0.,1.]

    for item in range(loop):
        S = Metropolis(S,temperature)
    magnetism = np.array([0, 0, 0], dtype=float)
    for n in range(4):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    magnetism += S[n,i,j,k]
    avmagnetism = magnetism/4/N**3
    return avmagnetism

Tlist = [Tinterval[0]+i*Tstep for i in range(int(Tinterval[1]/Tstep+1))]
magnetization = []
for T in Tlist:
    print('T = ',T)
    mlist = []
    for i in range(time):
        M = getEquilibrium(N,T,loop)
        m = np.linalg.norm(M)
        mlist.append(m)
        print('    magnetization:',m)
    magnetization.append(sum(mlist)/len(mlist))
print(Tlist,magnetization)
plt.plot(Tlist,magnetization,color='black')
plt.plot(Tlist,magnetization,'.',color='#d62728')
plt.xlabel('T/K')
plt.ylabel('M')
plt.title('Magnetization')
plt.show()