import matplotlib.pyplot as plt
import numpy as np
from classes2 import CelestialBody, TwoBody, ThreeBody


# Create planets
sun = CelestialBody(0, 0, 0, 333480.0)
mercury = CelestialBody(0.3871, 0.2056, 0.2408, 0.0553)
venus = CelestialBody(0.7233, 0.0068, 0.6152, 0.8150)
earth = CelestialBody(1.0, 0.0167086, 1.0, 1.0)
mars = CelestialBody(1.5237, 0.0934, 1.8809, 0.1074)
jupiter = CelestialBody(5.2028, 0.0483, 11.862, 317.89)
saturn = CelestialBody(9.5388, 0.0560, 29.456, 95.159)
uranus = CelestialBody(19.191, 0.0461, 84.07, 14.56)
neptune = CelestialBody(30.061, 0.0100, 164.81, 17.15)

#satellite = CelestialBody()

"""
Exercise 1
"""
"""
print("Exercise 1")

two_body = TwoBody(sun, earth, 0, 1.0, 1E-6, False)

x1, y1, u1, v1, ek1, ep1, t1 = two_body.euler_cromer()
x2, y2, u2, v2, ek2, ep2, t2 = two_body.runge_kutta()

plt.figure('exercise-1-1')
plt.title('Earth orbit with stepsize $\\tau = 10^{-6}$')
plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
plt.plot(x1, y1, label='Euler-Cromer')
plt.plot(x2, y2, label='Runge-Kutta')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=3)

plt.figure('exercise-1-2-1')
plt.title('Energy conservation for Euler-Cromer method')
plt.plot(t1, ek1, label='Kinetic energy')
plt.plot(t1, ep1, label='Potential energy')
plt.plot(t1, ek1 + ep1, label='Total energy')
plt.xlabel('Years [yr]')
plt.ylabel('E [J]')
plt.legend()

plt.figure('exercise-1-2-2')
plt.title('Energy conservation for Runge-Kutta method')
plt.plot(t2, ek2, label='Kinetic energy')
plt.plot(t2, ep2, label='Potential energy')
plt.plot(t2, ek2 + ep2, label='Total energy')
plt.xlabel('Years [yr]')
plt.ylabel('E [J]')
plt.legend()

plt.figure('exercise-1-3')
plt.title('Demonstration of Keplers Third Law')
plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
plt.plot(mercury.p**2, mercury.a**3, label='Mercury', marker='o', linestyle='none')
plt.plot(venus.p**2, venus.a**3, label='Venus', marker='o', linestyle='none')
plt.plot(earth.p**2, earth.a**3, label='Earth', marker='o', linestyle='none')
plt.plot(mars.p**2, mars.a**3, label='Mars', marker='o', linestyle='none')
plt.plot(jupiter.p**2, jupiter.a**3, label='Jupiter', marker='o', linestyle='none')
plt.plot(saturn.p**2, saturn.a**3, label='Saturn', marker='o', linestyle='none')
plt.plot(uranus.p**2, uranus.a**3, label='Uranus', marker='o', linestyle='none')
plt.plot(neptune.p**2, neptune.a**3, label='Neptune', marker='o', linestyle='none')
plt.plot(np.linspace(0, 27500, 100), np.linspace(0, 27500, 100))
plt.xlabel("$a^3$ [AU]")
plt.ylabel("$T^2$ [yr]")
plt.legend()

plt.show()

"""
"""
Exercise 2
"""
print("Exercise 2")

two_body = TwoBody(sun, mercury, 0, 0.2408, 1E-6, False)

"""
x, y, u, v, ek, ep, t = two_body.runge_kutta()

plt.figure('exercise-2-1-1')
plt.title('Mercury orbit with stepsize $\\tau = 10^{-6}$')
plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
plt.plot(x, y, label='Runge-Kutta')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=3)

plt.figure('exercise-2-1-2')
plt.title('Mercury orbit energy conservation using Runge-Kutta')
plt.plot(t, ek, label='Kinetic energy')
plt.plot(t, ep, label='Potential energy')
plt.plot(t, ek + ep, label='Total energy')
plt.xlabel('Years [yr]')
plt.ylabel('E [J]')
plt.legend()
"""

"""
x, y, u, v, ek, ep, t = two_body.runge_kutta(alpha=0.02)

plt.figure('exercise-2-2-1')
plt.title('Mercury orbit with stepsize $\\tau = 10^{-6}$ and $\\alpha = 0.02$')
plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
plt.plot(x, y, label='Runge-Kutta')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=3)
plt.show()
"""
N = 20
degrees = np.zeros(N)
alpha = np.linspace(0.0100, 0.0001, N)
for i in range(N):
    print("Iteration number", i + 1)

    x, y, u, v, ek, ep, t = two_body.runge_kutta(alpha=alpha[i])

    if np.arctan2(y[-1], x[-1]) < 0:
        degrees[i] = np.rad2deg((2 * np.pi) + np.arctan2(y[-1], x[-1]))
    else:
        degrees[i] = np.rad2deg(np.arctan2(y[-1], x[-1]))

    print("Alpha:", alpha[i])
    print("Precession:", degrees[i])


a, b = np.polyfit(alpha, degrees, 1)
print("Precession for Einsteins alpha:", (a * 1.1E-3) + b)

plt.figure('exercise-2-3')
plt.title('Precession of Mercury orbit over 0.2408 Earth years')
plt.plot(alpha, degrees, label='Numerical points', linestyle="none", marker="o")
plt.plot(np.linspace(0, np.amax(alpha), 100), a * np.linspace(0, np.amax(alpha), 100) + b, label='Regression line')
plt.xlabel("$\\alpha$")
plt.ylabel("Precession in degrees")
plt.legend()

plt.show()
"""
"""
"""
Exercise 3
"""
print("Exercise 3")


"""
degrees = np.zeros(10)
alfa = np.zeros(10)

for i in range(5):
    alfa[i] = 0.001 + (0.002 * i)
    x, y, u, v, ek, ep, t = two_body.runge_kutta(alpha=alfa[i])
    degrees[i] = np.rad2deg(np.arctan2(y[-1], x[-1]))

    print("Turn:", i + 1)
    print("Precession in degrees:", degrees[i])
    print("Alfa:", alfa[i])

for i in range(5):
    alfa[5 + i] = 0.0001 + (0.0002 * i)
    x, y, u, v, ek, ep, t = two_body.runge_kutta(alpha=alfa[5 + i])
    degrees[5 + i] = np.rad2deg(np.arctan2(y[-1], x[-1]))

    print("Turn:", i + 6)
    print("Precession in degrees:", degrees[5 + i])
    print("Alfa:", alfa[5 + i])

# Perform regression (x-axis = alfa, y-axis = degrees)
a, b = np.polyfit(alfa, degrees, 1)

print("Precession in degrees for alfa = 1.1E-8 times 5.0E4:", (a * 1.1E-8 * 5.0E4) + b)

plt.figure('Plot')
plt.title('Precession of Mercury orbit over 0.2408 Earth years')
plt.plot(alfa, degrees, label='Numerical points', linestyle="none", marker="o")
plt.plot(np.linspace(0, np.amax(alfa), 100), a * np.linspace(0, np.amax(alfa), 100) + b, label='Regression line')
plt.xlabel("$\\alpha$")
plt.ylabel("Precession in degrees")
plt.legend()
plt.show()

"""
"""
plt.figure('exercise-2')
plt.title('Demonstration of Keplers Third Law')
plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
plt.plot(mercury.p**2, mercury.a**3, label='Mercury', marker='o', linestyle='none')
plt.plot(venus.p**2, venus.a**3, label='Venus', marker='o', linestyle='none')
plt.plot(earth.p**2, earth.a**3, label='Earth', marker='o', linestyle='none')
plt.plot(mars.p**2, mars.a**3, label='Mars', marker='o', linestyle='none')
plt.plot(jupiter.p**2, jupiter.a**3, label='Jupiter', marker='o', linestyle='none')
plt.plot(saturn.p**2, saturn.a**3, label='Saturn', marker='o', linestyle='none')
plt.plot(uranus.p**2, uranus.a**3, label='Uranus', marker='o', linestyle='none')
plt.plot(neptune.p**2, neptune.a**3, label='Neptune', marker='o', linestyle='none')
plt.plot(np.arange(0, 30000, 1), np.arange(0, 30000, 1))
plt.xlabel("$a^3$")
plt.ylabel("$T^2$")
plt.legend()

plt.show()

plt.figure('exercise-2')
plt.title('Total energy Earth')
#plt.plot(t, ek + ep, label='Earth Euler')
plt.plot(r_t, r_ek + r_ep, label='Earth Kutta')
plt.legend()

plt.show()

# Three body motion
three_body = ThreeBody(sun, earth, mars, 0, period, stepsize)
x1, y1, x2, y2, u1, v1, u2, v2, ek1, ep1, ek2, ep2, t = three_body.euler_cromer()
r_u1, r_v1, r_u2, r_v2, r_x1, r_y1, r_x2, r_y2, r_ek1, r_ek2, r_ep1, r_ep2, r_t = three_body.runge_kutta()

plt.figure('exercise-3')
plt.title('Orbit in three body motion')
plt.plot(0, 0, label='The sun', marker='o', linestyle='none')
plt.plot(x1, y1, label='Earth Euler')
plt.plot(x2, y2, label='Mars Euler')
plt.plot(r_x1, r_y1, label='Earth Kutta')
plt.plot(r_x2, r_y2, label='Mars Kutta')
plt.legend()

plt.figure('exercise-4')
plt.title('Total energy Earth')
plt.plot(t, ek1 + ep1, label="Earth Euler")
plt.plot(r_t, r_ek1 + r_ep1, label="Earth Kutta")
plt.legend()

plt.figure('exercise-5')
plt.title('Total energy Mars')
plt.plot(t, ek2 + ep2, label="Mars Euler")
plt.plot(r_t, r_ek2 + r_ep2, label="Mars Kutta")
plt.legend()

plt.show()
"""