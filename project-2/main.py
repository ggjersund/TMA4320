import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from classes import Protein

# Set default plot values
plt.rcParams['axes.facecolor'] = '#e6e6e6'
plt.rcParams['grid.color'] = '#ffffff'
plt.rcParams['grid.linestyle'] = '-'


# Exercise 1
def oppg_1():
    N = 10
    T = 100

    # Create the protein object
    protein = Protein(N, T)

    # Start subplot - 1 twist

    plt.figure('exercise-1')
    plt.suptitle('Protein folding')

    plt.subplot(131)

    # Create vectors for plotting the protein
    x1 = []
    y1 = []

    for i in range(N):
        x1.append(int(protein.matrix[i][0]))
        y1.append(int(protein.matrix[i][1]))

    # Plot the unfolded protein
    plt.grid()
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.xticks(np.arange(0, N, 1))
    plt.yticks(np.arange(0, N, 1))
    plt.plot(x1, y1)
    plt.plot(x1, y1, color='#000000', linestyle='-', marker='o', markerfacecolor='#aeaeae', markersize='12')

    for i in range(1, N + 1):
        plt.annotate(i, (x1[i - 1], y1[i - 1]), xytext=(x1[i - 1] + 0.1, y1[i - 1] + 0.1))

    plt.subplot(132)

    # Perform the first twist
    protein.twist()

    x2 = []
    y2 = []

    for i in range(N):
        x2.append(int(protein.matrix[i][0]))
        y2.append(int(protein.matrix[i][1]))

    # Plot the once-folded protein
    plt.grid()
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.xticks(np.arange(0, N, 1))
    plt.yticks(np.arange(0, N, 1))
    plt.plot(x2, y2)
    plt.plot(x2, y2, color='#000000', linestyle='-', marker='o', markerfacecolor='#aeaeae', markersize='12')

    for i in range(1, N + 1):
        plt.annotate(i, (x2[i - 1], y2[i - 1]), xytext=(x2[i - 1] + 0.1, y2[i - 1] + 0.1))

    plt.subplot(133)

    # Perform the second twist
    protein.twist()

    x3 = []
    y3 = []
    for i in range(N):
        x3.append(int(protein.matrix[i][0]))
        y3.append(int(protein.matrix[i][1]))

    # Plot the twice-folded protein
    plt.grid()
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.xticks(np.arange(0, N, 1))
    plt.yticks(np.arange(0, N, 1))
    plt.plot(x3, y3)
    plt.plot(x3, y3, color='#000000', linestyle='-', marker='o', markerfacecolor='#aeaeae', markersize='12')

    for i in range(1, N + 1):
        plt.annotate(i, (x3[i - 1], y3[i - 1]), xytext=(x3[i - 1] + 0.1, y3[i - 1] + 0.1))

    plt.show()


# Exercise 2.1
def oppg_2_1():
    T_step = 5# Increase the temp by this amount each iteration
    N = 15
    T = 10**(-6)  # Define temperature CLOSE to zero to avoid division by zero
    s = 0.003
    d_max = 10000
    T_max = 1500
    N_T = T_max // T_step

    # Matrix with all energies for all twists and temp
    epsilon = np.zeros((N_T, d_max))

    # Vector with mean energy for each temp
    e_mean = np.zeros(N_T)

    # Calculate mean energy for temp close to zero
    protein = Protein(N, T)
    for i in range(0, d_max):
        protein.twist()
        epsilon[0][i] = protein.energy()
    e_mean[0] = np.mean(epsilon[0][:d_max])

    # Calculate mean energy for all other temps up to T_max
    for j in range(1, N_T):
        protein = Protein(N, j * T_step)
        d = int(np.ceil(d_max * np.exp(-s * (j * T_step))))
        for i in range(0, d):
            protein.twist()
            epsilon[j][i] = protein.energy()
        e_mean[j] = np.mean(epsilon[j][:d])

    # Create temp vector for plotting the mean energy
    x = np.arange(0, T_max, T_step)

    # Plot the mean energy
    plt.figure('exercise-2-1')
    plt.title('Mean energy')
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('$<E>$')
    plt.plot(x, e_mean, color='#000000', linestyle='-')
    plt.show()


# Exercise 2.2
def oppg_2_2():
    N = 15
    T_0 = 10**(-6)
    T_500 = 500
    d = 5000
    x = np.arange(0, d)

    protein1 = Protein(N, T_0)
    protein2 = Protein(N, T_500)

    energy1 = np.zeros(d)
    energy2 = np.zeros(d)

    for i in range(d):
        protein1.twist()
        protein2.twist()
        energy1[i] = protein1.energy()
        energy2[i] = protein2.energy()

    plt.figure('exercise-2-2')
    plt.suptitle('Binding energy')

    # Plot for T = 0 Kelvin
    plt.subplot(121)
    plt.title('$T = 0$')
    plt.grid()
    plt.xlabel('$\log$ # twists')
    plt.ylabel('E')
    plt.semilogx(x, energy1, color='#000000', linestyle='-')

    # Plot for T = 500 Kelvin
    plt.subplot(122)
    plt.title('$T = 500$')
    plt.xlabel('# twists')
    plt.ylabel('E')
    plt.grid()
    plt.plot(x, energy2, color='#000000', linestyle='-', linewidth=0.7)

    plt.show()


# Exercise 2.4
def oppg_2_4():
    N = 15
    T = 10**(-6)
    d = 10000

    # Number of protein plots
    calculations = 10

    mean_energy = np.zeros(calculations)
    for i in range(len(mean_energy)):
        protein = Protein(N, T)
        energy = np.zeros(d)

        for j in range(d):
            protein.twist()
            energy[j] = protein.energy()

        mean_energy[i] = np.mean(energy)

    plt.figure('exercise-2-4')
    plt.title('Mean energy at $T = 0$')
    plt.grid()
    plt.xlabel('Calculation #')
    plt.ylabel('$<$E$>$')
    plt.xticks(np.arange(0, calculations, 1))
    plt.plot(np.arange(calculations), mean_energy, color='#000000', markerfacecolor='#000000', linestyle='None', marker='o')
    plt.show()


# Exercise 3
def oppg_3():
    T_step = 5  # Increase the temp by this amount each iteration
    N = 15
    T = 10 ** (-6)  # Define temperature CLOSE to zero to avoid division by zero
    s = 0.003
    d_max = 10000
    T_max = 1500
    N_T = T_max // T_step

    # Matrix with all diameters for all twists and temps
    length = np.zeros((N_T, d_max))

    # Vector with mean energy for each temp
    L_mean = np.zeros(N_T)

    # Calculate mean diameter for temp close to zero
    protein = Protein(N, T)
    for i in range(0, d_max):
        protein.twist()
        length[0][i] = protein.diameter()
    L_mean[0] = np.mean(length[0][:d_max])

    # Calculate mean diameter for all other temps up to T_max
    for j in range(1, N_T):
        print(j)
        protein = Protein(N, j * T_step)
        d = int(np.ceil(d_max * np.exp(-s * (j * T_step))))
        for i in range(0, d):
            protein.twist()
            length[j][i] = protein.diameter()
        L_mean[j] = np.mean(length[j][:d])

    # Create temp vector for plotting the mean diameter
    x = np.arange(0, T_max, T_step)

    # Plot the mean diameter
    plt.figure('exercise-3')
    plt.title('Mean diameter')
    plt.grid(color='#ffffff', linestyle='-')
    plt.xlabel('T')
    plt.ylabel('$<L>$')
    plt.plot(x, L_mean, color='#000000', linestyle='-')
    plt.show()


# Exercise 4.1
def oppg_4_1 ():
    T_step = -30  # increase the temp by this amount each iteration
    N = 15
    T_max = 1500
    d = 600
    N_T = (T_max // abs(T_step)) * d
    count = 0

    # Vector with energy
    energy = np.zeros(N_T)

    # Calculate energy
    protein = Protein(N, T_max)
    for j in range(T_max, -T_step, T_step):
        print(j)
        protein.change_temp(j)
        for i in range(0, d):
            protein.twist()
            energy[count] = protein.energy()
            count += 1

    protein.change_temp(10**(-6))
    for i in range(0, d):
        protein.twist()
        energy[count] = protein.energy()
        count += 1

    # Create temp vector for plotting the energy
    x = np.arange(0, N_T)

    # Plot the energy
    plt.figure('exercise-4-1')
    plt.title('Protein energy')
    plt.grid()
    plt.xlabel('# twists')
    plt.ylabel('Energy')
    labels = []
    for i in range(0, N_T, d):
        if (i % (N_T / 10)) == 0:
            labels.append(str(i))
        else:
            labels.append("")

    plt.xticks(np.arange(0, N_T, d), labels)
    plt.plot(x, energy, color='#000000', linestyle='-', linewidth=0.3)
    plt.show()


# Exercise 4.2
def oppg_4_2 ():
    T_step = -30  # increase the temp by this amount each iteration
    N = 30
    T_max = 1500
    d = 600
    N_T = (T_max // abs(T_step))
    count = 0

    # Matrix with all energies for all twists and temp
    epsilon = np.zeros((N_T, d))

    # Vector with mean energy for each temp
    e_mean = np.zeros(N_T)

    # Calculate energy
    protein = Protein(N, T_max)
    for j in range(T_max, -T_step, T_step):
        print(j)
        protein.change_temp(j)
        for i in range(0, d):
            protein.twist()
            epsilon[count][i] = protein.energy()
        e_mean[count] = np.mean(epsilon[count])
        count += 1

    protein.change_temp(10 ** (-6))
    for i in range(0, d):
        protein.twist()
        epsilon[N_T -1] = protein.energy()
    e_mean[N_T - 1] = np.mean(epsilon[N_T - 1])
    e_mean = e_mean[::-1]

    # Create temp vector for plotting the mean energy
    x = np.arange(0, T_max, -T_step)

    # Plot the mean energy
    plt.figure('exercise-4-2')
    plt.title('Mean energy')
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('$<$E$>$')
    plt.plot(x, e_mean, color='#000000', linestyle='-')
    plt.show()


# Exercise 4.3
def oppg_4_3(return_protein):
    T_step = -30  # increase the temp by this amount each iteration
    N = 15
    T_max = 1500
    d = 600
    N_T = (T_max // abs(T_step))
    count = 0

    # Matrix with all diameters for all twists and temp
    length = np.zeros((N_T, d))

    # Vector with mean energy for each temp
    L_mean = np.zeros(N_T)

    # Calculate energy
    protein = Protein(N, T_max)
    for j in range(T_max, -T_step, T_step):
        print(j)
        protein.change_temp(j)
        for i in range(0, d):
            protein.twist()
            length[count][i] = protein.diameter()
        L_mean[count] = np.mean(length[count])
        count += 1

    protein.change_temp(10 ** (-6))
    for i in range(0, d):
        protein.twist()
        length[N_T - 1] = protein.diameter()
    L_mean[N_T - 1] = np.mean(length[N_T - 1])
    L_mean = L_mean[::-1]

    # Create temp vector for plotting the mean diameter
    x = np.arange(0, T_max, -T_step)

    # Plot the mean diameter
    plt.figure('exercise-4-3')
    plt.title('Mean diameter')
    plt.grid()
    plt.xlabel('T')
    plt.ylabel('$<$L$>$')
    plt.plot(x, L_mean, color='#000000', linestyle='-')
    plt.show()

    if return_protein:
        return protein, N


# Exercise 4.4
def oppg_4_4():
    # look at a protein beeing cooled from 1500K to 0K and plot the mean diameter ..
    # .. for each temperature.
    protein, N = oppg_4_3(True)

    # Create vectors for plotting the protein
    x = []
    y = []
    for i in range(N):
        x.append(int(protein.matrix[i][0]))
        y.append(int(protein.matrix[i][1]))

    # Draw the protein at the end, Temp near zero
    plt.figure('exercise-4-4')
    plt.title('Protein')
    plt.grid()
    plt.xlim(0, N - 1)
    plt.ylim(0, N - 1)
    plt.xticks(np.arange(0, N, 1))
    plt.yticks(np.arange(0, N, 1))
    plt.plot(x, y)
    plt.plot(x, y, color='#000000', linestyle='-', marker='o', markerfacecolor='#aeaeae', markersize='12')

    for i in range(1, N + 1):
        plt.annotate(i, (x[i - 1], y[i - 1]), xytext=(x[i - 1] + 0.1, y[i - 1] + 0.1))

    plt.show()


oppg_4_2()
