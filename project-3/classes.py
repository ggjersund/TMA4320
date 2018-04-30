import numpy as np


class CelestialBody:

    # Initialize variables
    def __init__(self, a, e, p, m):
        self.a = a
        self.e = e
        self.p = p
        self.m = m


class TwoBody:

    # Initialize variables
    def __init__(self, center, satellite, t_i, t_f, t_step):

        self.center = center
        self.satellite = satellite
        self.t_i = t_i
        self.t_f = t_f
        self.t_step = t_step

        # Calculate initial Velocity
        self.vi = np.sqrt((4 * np.pi**2 *(self.center.m / 333480) * (1 - self.satellite.e) / (self.satellite.a * (1 + self.satellite.e))) * (1+self.satellite.m/self.center.m))
        self.ui = 0.0

        # Calculate initial position
        self.xi = self.satellite.a * (1 + self.satellite.e)
        self.yi = 0.0

        # Print initial conditions
        print("Initial conditions for two body motion")
        print("Initial velocity for the satellite:\t", self.vi)
        print("Initial position for the satellite:\t", self.xi, "\n")

    @staticmethod
    def euler_rhs(x, y, u, v, m):
        r = np.sqrt(x**2 + y**2)
        dxdt = u
        dydt = v
        dudt = - 4 * np.pi**2 * x / r**3
        dvdt = - 4 * np.pi**2 * y / r**3
        ek = 0.5 * m * (u**2 + v**2)
        ep = - (4 * np.pi**2 * m / r)

        return dxdt, dydt, dudt, dvdt, ek, ep

    def euler_cromer(self):

        # Configuration
        N = len(np.arange(0, self.t_f, self.t_step))
        t = np.zeros(N)

        # Satellite
        x = np.zeros(N)
        y = np.zeros(N)
        u = np.zeros(N)
        v = np.zeros(N)
        ek = np.zeros(N)
        ep = np.zeros(N)

        # Set initial conditions
        x[0] = self.xi
        y[0] = self.yi
        u[0] = self.ui
        v[0] = self.vi
        ek[0] = 0.5 * self.satellite.m * (u[0]**2 + v[0]**2)
        ep[0] = - (4 * np.pi**2 * self.satellite.m / np.sqrt(x[0]**2 + y[0]**2))

        # Initial time
        t[0] = 0.0

        # Iterate time
        for n in range(N - 1):
            (dX, dY, dU, dV, ek[n + 1], ep[n + 1]) = self.euler_rhs(x[n], y[n], u[n], v[n], self.satellite.m)
            u[n + 1] = u[n] + (self.t_step * dU)
            v[n + 1] = v[n] + (self.t_step * dV)
            # Worse version is when u[n] * dt instead of u[n+1] * dt
            x[n + 1] = x[n] + (u[n + 1] * self.t_step)
            y[n + 1] = y[n] + (v[n + 1] * self.t_step)
            t[n + 1] = n * self.t_step

        # Return result
        return x, y, u, v, ek, ep, t

    def runge_kutta(self):
        C = 4 * (np.pi ** 2) * self.center.m / 333480  # mass times the constant G in AU
        N = int(self.t_f / self.t_step) + 1  # number of itertions
        alpha = 0

        def H(X, Y):  # acceleration in x direction
            return (-C * (X / ((np.sqrt(X ** 2 + Y ** 2)) ** 3))) * (1 + alpha /(X ** 2 + Y ** 2))

        def I(X, Y):  # acceleration in x direction
            return (-C * (Y / ((np.sqrt(X ** 2 + Y ** 2)) ** 3))) * (1 + alpha /(X ** 2 + Y ** 2))

        X_4RK = np.zeros(N)
        Y_4RK = np.zeros(N)
        U_4RK = np.zeros(N)
        V_4RK = np.zeros(N)
        EK_4RK = np.zeros(N)
        EP_4RK = np.zeros(N)
        T_4RK = np.zeros(N)

        X_4RK[0] = self.xi
        Y_4RK[0] = self.yi
        V_4RK[0] = self.vi
        EK_4RK[0] = 0.5 * self.satellite.m * (U_4RK[0] ** 2 + V_4RK[0] ** 2)
        EP_4RK[0] = -4 * self.satellite.m * np.pi ** 2 / np.sqrt(X_4RK[0] ** 2 + Y_4RK[0] ** 2)
        T_4RK[0] = 0

        for n in range(N - 1):
            k_x1 = self.t_step * U_4RK[n]
            k_y1 = self.t_step * V_4RK[n]
            k_u1 = self.t_step * H(X_4RK[n], Y_4RK[n])
            k_v1 = self.t_step * I(X_4RK[n], Y_4RK[n])

            k_x2 = self.t_step * (U_4RK[n] + k_u1 / 2)
            k_y2 = self.t_step * (V_4RK[n] + k_v1 / 2)
            k_u2 = self.t_step * H(X_4RK[n] + k_x1 / 2, Y_4RK[n] + k_y1 / 2)
            k_v2 = self.t_step * I(X_4RK[n] + k_x1 / 2, Y_4RK[n] + k_y1 / 2)

            k_x3 = self.t_step * (U_4RK[n] + k_u2 / 2)
            k_y3 = self.t_step * (V_4RK[n] + k_v2 / 2)
            k_u3 = self.t_step * H(X_4RK[n] + k_x2 / 2, Y_4RK[n] + k_y2 / 2)
            k_v3 = self.t_step * I(X_4RK[n] + k_x2 / 2, Y_4RK[n] + k_y2 / 2)

            k_x4 = self.t_step * (U_4RK[n] + k_u3)
            k_y4 = self.t_step * (V_4RK[n] + k_v3)
            k_u4 = self.t_step * H(X_4RK[n] + k_x3, Y_4RK[n] + k_y3)
            k_v4 = self.t_step * I(X_4RK[n] + k_x3, Y_4RK[n] + k_y3)

            X_4RK[n + 1] = X_4RK[n] + k_x1 / 6 + k_x2 / 3 + k_x3 / 3 + k_x4 / 6
            Y_4RK[n + 1] = Y_4RK[n] + k_y1 / 6 + k_y2 / 3 + k_y3 / 3 + k_y4 / 6
            U_4RK[n + 1] = U_4RK[n] + k_u1 / 6 + k_u2 / 3 + k_u3 / 3 + k_u4 / 6
            V_4RK[n + 1] = V_4RK[n] + k_v1 / 6 + k_v2 / 3 + k_v3 / 3 + k_v4 / 6
            EK_4RK[n + 1] = 0.5 * self.satellite.m * (U_4RK[n + 1] ** 2 + V_4RK[n + 1] ** 2)
            EP_4RK[n + 1] = -4 * self.satellite.m * np.pi ** 2 / np.sqrt(X_4RK[n + 1] ** 2 + Y_4RK[n + 1] ** 2)
            T_4RK[n + 1] = (n+1) * self.t_step

        return X_4RK, Y_4RK, U_4RK, V_4RK, EK_4RK, EP_4RK, T_4RK

    def oppg_3(self, transfers, r_max):
        C = 4 * (np.pi ** 2) * self.center.m / 333480
        N = int(self.t_f / self.t_step) + 1
        AU = 1.5 * 10**11
        YR = 3.2 * 10**7
        M = 6 * 10**24
        G = C / self.center.m

        a = (r_max + r_min)/2
        e = (r_max / a)-1
        V_1 = np.sqrt((4*np.pi**2/333480) * ((1 + e) / (a * (1 - e))) * (1 + (self.satellite.m / self.center.m)))
        V_2 = np.sqrt((4*np.pi**2/333480) * (1 / r_max) * (1 + (self.satellite.m / self.center.m)))
        IS_LAUNCHED = False
        IS_LAUNCHED_TWICE = False
        P = np.sqrt(a**3 * 333480)/2


        def H(X, Y):  # acceleration in x direction
            return -C * (X / ((np.sqrt(X ** 2 + Y ** 2)) ** 3))

        def I(X, Y):  # acceleration in x direction
            return -C * (Y / ((np.sqrt(X ** 2 + Y ** 2)) ** 3))

        X_4RK = np.zeros(N)
        Y_4RK = np.zeros(N)
        U_4RK = np.zeros(N)
        V_4RK = np.zeros(N)
        EK_4RK = np.zeros(N)
        EP_4RK = np.zeros(N)
        T_4RK = np.zeros(N)

        X_4RK[0] = self.xi
        Y_4RK[0] = self.yi
        V_4RK[0] = self.vi
        EK_4RK[0] = 0.5 * self.satellite.m * (U_4RK[0] ** 2 + V_4RK[0] ** 2)
        EP_4RK[0] = -4 * self.satellite.m * self.center.m * np.pi**2 / (np.sqrt(X_4RK[0] ** 2 + Y_4RK[0] ** 2) * 333480)
        T_4RK[0] = 0

        for n in range(N - 1):
            if n*self.t_step > self.satellite.p and not IS_LAUNCHED:
                X_HAT = U_4RK[n-1] / np.sqrt(U_4RK[n-1] ** 2 + V_4RK[n-1] ** 2)
                Y_HAT = V_4RK[n-1] / np.sqrt(U_4RK[n-1] ** 2 + V_4RK[n-1] ** 2)
                U_4RK[n] = X_HAT * V_1
                V_4RK[n] = Y_HAT * V_1
                IS_LAUNCHED = True
                print("if 1, time: ", T_4RK[n])


            if ((n*self.t_step > (self.satellite.p + P)) and IS_LAUNCHED and not IS_LAUNCHED_TWICE):
                X_HAT = U_4RK[n - 1] / np.sqrt(U_4RK[n - 1] ** 2 + V_4RK[n - 1] ** 2)
                Y_HAT = V_4RK[n - 1] / np.sqrt(U_4RK[n - 1] ** 2 + V_4RK[n - 1] ** 2)
                U_4RK[n] = X_HAT * V_2
                V_4RK[n] = Y_HAT * V_2
                IS_LAUNCHED_TWICE = True
                print("if 2, time: ", T_4RK[n])




            k_x1 = self.t_step * U_4RK[n]
            k_y1 = self.t_step * V_4RK[n]
            k_u1 = self.t_step * H(X_4RK[n], Y_4RK[n])
            k_v1 = self.t_step * I(X_4RK[n], Y_4RK[n])

            k_x2 = self.t_step * (U_4RK[n] + k_u1 / 2)
            k_y2 = self.t_step * (V_4RK[n] + k_v1 / 2)
            k_u2 = self.t_step * H(X_4RK[n] + k_x1 / 2, Y_4RK[n] + k_y1 / 2)
            k_v2 = self.t_step * I(X_4RK[n] + k_x1 / 2, Y_4RK[n] + k_y1 / 2)

            k_x3 = self.t_step * (U_4RK[n] + k_u2 / 2)
            k_y3 = self.t_step * (V_4RK[n] + k_v2 / 2)
            k_u3 = self.t_step * H(X_4RK[n] + k_x2 / 2, Y_4RK[n] + k_y2 / 2)
            k_v3 = self.t_step * I(X_4RK[n] + k_x2 / 2, Y_4RK[n] + k_y2 / 2)

            k_x4 = self.t_step * (U_4RK[n] + k_u3)
            k_y4 = self.t_step * (V_4RK[n] + k_v3)
            k_u4 = self.t_step * H(X_4RK[n] + k_x3, Y_4RK[n] + k_y3)
            k_v4 = self.t_step * I(X_4RK[n] + k_x3, Y_4RK[n] + k_y3)

            X_4RK[n + 1] = X_4RK[n] + k_x1 / 6 + k_x2 / 3 + k_x3 / 3 + k_x4 / 6
            Y_4RK[n + 1] = Y_4RK[n] + k_y1 / 6 + k_y2 / 3 + k_y3 / 3 + k_y4 / 6
            U_4RK[n + 1] = U_4RK[n] + k_u1 / 6 + k_u2 / 3 + k_u3 / 3 + k_u4 / 6
            V_4RK[n + 1] = V_4RK[n] + k_v1 / 6 + k_v2 / 3 + k_v3 / 3 + k_v4 / 6
            EK_4RK[n + 1] = 0.5 * self.satellite.m * (U_4RK[n + 1]**2 + V_4RK[n + 1]**2)
            EP_4RK[n + 1] = -4 * np.pi**2 * self.satellite.m * self.center.m / ( np.sqrt(X_4RK[n + 1]**2 + Y_4RK[n + 1]**2) * 333480)
            T_4RK[n + 1] = (n+1) * self.t_step

        return X_4RK * AU, Y_4RK * AU, U_4RK * AU/YR, V_4RK * AU/YR, EK_4RK * M *(AU/YR)**2, EP_4RK * M *AU**2/(YR**2), T_4RK*YR


class ThreeBody:

    # Initialize variables
    def __init__(self, center, satellite_1, satellite_2, t_i, t_f, t_step):
        self.center = center
        self.satellite_1 = satellite_1
        self.satellite_2 = satellite_2
        self.t_i = t_i
        self.t_f = t_f
        self.t_step = t_step

        # Calculate initial velocity for satellite 1
        self.v1i = np.sqrt(4 * np.pi**2 * (1 - self.satellite_1.e) / (self.satellite_1.a * (1 + self.satellite_1.e)))
        self.u1i = 0.0

        # Calculate initial velocity for satellite 2
        self.v2i = np.sqrt(4 * np.pi**2 * (1 - self.satellite_2.e) / (self.satellite_2.a * (1 + self.satellite_2.e)))
        self.u2i = 0.0

        # Calculate initial position for satellite 1
        self.x1i = self.satellite_1.a * (1 + self.satellite_1.e)
        self.y1i = 0.0

        # Calculate initial position for satellite 2
        self.x2i = self.satellite_2.a * (1 + self.satellite_2.e)
        self.y2i = 0.0

        # Print initial conditions
        print("Initial conditions for two body motion")
        print("Initial velocity for Satellite 1:\t", self.v1i)
        print("Initial position for Satellite 1:\t", self.x1i)
        print("Initial velocity for Satellite 2:\t", self.v2i)
        print("Initial position for Satellite 2:\t", self.x2i, "\n")

    @staticmethod
    def euler_rhs(x1, x2, y1, y2, u1, u2, v1, v2, mc, m1, m2):
        r1 = np.sqrt(x1**2 + y1**2)
        r2 = np.sqrt(x2**2 + y2**2)
        r12 = np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))

        u1i = u1
        v1i = v1

        u2i = u2
        v2i = v2

        dudt1 = - (4 * np.pi**2 * x1 / r1**3) - (4 * np.pi**2 * (m2 / mc) * (x1 - x2) / r12**3)
        dvdt1 = - (4 * np.pi**2 * y1 / r1**3) - (4 * np.pi**2 * (m2 / mc) * (y1 - y2) / r12**3)

        dudt2 = - (4 * np.pi**2 * x2 / r2**3) - (4 * np.pi**2 * (m1 / mc) * (x2 - x1) / r12**3)
        dvdt2 = - (4 * np.pi**2 * y2 / r2**3) - (4 * np.pi**2 * (m1 / mc) * (y2 - y1) / r12**3)

        ek1 = 0.5 * m1 * (u1**2 + v1**2)
        ep1_c1 = - (4 * np.pi**2 * m1 / r1)
        ep1_12 = - (4 * np.pi**2 * m1 * m2 / (r12 * mc))
        ep1 = ep1_c1 + ep1_12

        ek2 = 0.5 * m2 * (u2**2 + v2**2)
        ep2_c2 = - (4 * np.pi**2 * m2 / r2)
        ep2_12 = - (4 * np.pi**2 * m2 * m1 / (r12 * mc))
        ep2 = ep2_c2 + ep2_12

        return u1i, v1i, u2i, v2i, dudt1, dvdt1, dudt2, dvdt2, ek1, ep1, ek2, ep2

    def euler_cromer(self):

        # Configuration
        N = len(np.arange(0, self.t_f, self.t_step))
        t = np.zeros(N)

        # Satellite 1
        x1 = np.zeros(N)
        y1 = np.zeros(N)
        u1 = np.zeros(N)
        v1 = np.zeros(N)
        ek1 = np.zeros(N)
        ep1 = np.zeros(N)

        # Satellite 2
        x2 = np.zeros(N)
        y2 = np.zeros(N)
        u2 = np.zeros(N)
        v2 = np.zeros(N)
        ek2 = np.zeros(N)
        ep2 = np.zeros(N)

        # Set initial conditions
        # Satellite 1
        x1[0] = self.x1i
        y1[0] = self.y1i
        u1[0] = self.u1i
        v1[0] = self.v1i

        ek1[0] = 0.5 * self.satellite_1.m * (u1[0]**2 + v1[0]**2)

        ep1_c1 = - (4 * np.pi**2 * self.satellite_1.m / np.sqrt(x1[0]**2 + y1[0]**2))
        ep1_12 = - (4 * np.pi**2 * self.satellite_1.m * self.satellite_2.m / (np.sqrt((x1[0] - x2[0])**2 + (y1[0] - y2[0])**2) * self.center.m))
        ep1[0] = ep1_c1 + ep1_12

        # Satellite 2
        x2[0] = self.x2i
        y2[0] = self.y2i
        u2[0] = self.u2i
        v2[0] = self.v2i
        ek2[0] = 0.5 * self.satellite_2.m * (u2[0]**2 + v2[0]**2)
        ep2_c2 = - (4 * np.pi**2 * self.satellite_2.m / np.sqrt(x2[0]**2 + y2[0]**2))
        ep2_12 = - (4 * np.pi**2 * self.satellite_2.m * self.satellite_1.m / (np.sqrt((x1[0] - x2[0])**2 + (y1[0] - y2[0])**2) * self.center.m))
        ep2[0] = ep2_c2 + ep2_12

        # Initial time
        t[0] = 0.0

        for n in range(N - 1):

            # Calculate
            (u1i,
             v1i,
             u2i,
             v2i,
             dudt1,
             dvdt1,
             dudt2,
             dvdt2,
             ek1[n + 1],
             ep1[n + 1],
             ek2[n + 1],
             ep2[n + 1]) = self.euler_rhs(x1[n],
                                          x2[n],
                                          y1[n],
                                          y2[n],
                                          u1[n],
                                          u2[n],
                                          v1[n], v2[n],
                                          self.center.m,
                                          self.satellite_1.m,
                                          self.satellite_2.m)

            # Satellite 1
            u1[n + 1] = u1i + (dudt1 * self.t_step)
            v1[n + 1] = v1i + (dvdt1 * self.t_step)
            x1[n + 1] = x1[n] + (u1[n + 1] * self.t_step)
            y1[n + 1] = y1[n] + (v1[n + 1] * self.t_step)

            # Satellite 2
            u2[n + 1] = u2i + (dudt2 * self.t_step)
            v2[n + 1] = v2i + (dvdt2 * self.t_step)
            x2[n + 1] = x2[n] + (u2[n + 1] * self.t_step)
            y2[n + 1] = y2[n] + (v2[n + 1] * self.t_step)

            t[n + 1] = n * self.t_step

        return x1, y1, x2, y2, u1, v1, u2, v2, ek1, ep1, ek2, ep2, t

    def runge_kutta(self):

        C = 4 * (np.pi ** 2) # mass of planet E times the constant G
        C_inter1 = 4 * (np.pi ** 2) * (self.satellite_2.m / self.center.m)  # mass of planet J times the constant G (interaction)
        C_inter2 = 4 * (np.pi ** 2) * (self.satellite_1.m / self.center.m)  # mass of planet E times the constant G (interaction)
        N = int(self.t_f / self.t_step) + 1  # number of itertions

        def HE(X1, Y1, X2, Y2):  # acceleration in x direction for satellite 1
            return -C * X1 / ((np.sqrt(X1 ** 2 + Y1 ** 2)) ** 3) + C_inter1 * ((X2 - X1) / (
                np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)) ** 3)

        def IE(X1, Y1, X2, Y2):  # acceleration in y direction for satellite 1
            return -C * Y1 / ((np.sqrt(X1 ** 2 + Y1 ** 2)) ** 3) + C_inter1 * (
                    (Y2 - Y1) / (np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)) ** 3)

        def HJ(X1, Y1, X2, Y2):  # acceleration in x direction for satellite 2
            return -C * X2 / ((np.sqrt(X2 ** 2 + Y2 ** 2)) ** 3) - C_inter2 * ((X2 - X1) / (
                np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)) ** 3)

        def IJ(X1, Y1, X2, Y2):  # acceleration in y direction for satellite 2
            return -C * Y2 / ((np.sqrt(X2 ** 2 + Y2 ** 2)) ** 3) - C_inter2 * (
                    (Y2 - Y1) / (np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)) ** 3)

        X1_4RK = np.zeros(N)
        Y1_4RK = np.zeros(N)
        U1_4RK = np.zeros(N)
        V1_4RK = np.zeros(N)

        X2_4RK = np.zeros(N)
        Y2_4RK = np.zeros(N)
        U2_4RK = np.zeros(N)
        V2_4RK = np.zeros(N)

        EK1_4RK = np.zeros(N)
        EK2_4RK = np.zeros(N)
        EP1_4RK = np.zeros(N)
        EP2_4RK = np.zeros(N)
        T_4RK = np.zeros(N)

        X1_4RK[0] = self.x1i
        Y1_4RK[0] = self.y1i
        V1_4RK[0] = self.v1i

        X2_4RK[0] = self.x2i
        Y2_4RK[0] = self.y2i
        V2_4RK[0] = self.v2i

        EK1_4RK[0] = 0.5 * self.satellite_1.m * (U1_4RK[0]**2 + V1_4RK[0]**2)
        EK2_4RK[0] = 0.5 * self.satellite_2.m * (U2_4RK[0]**2 + V2_4RK[0]**2)

        EP1_4RK[0] = -((4 * np.pi**2 * self.satellite_1.m / np.sqrt(X1_4RK[0]**2 + Y1_4RK[0]**2)) + (
                4 * np.pi**2 * self.satellite_1.m * self.satellite_2.m / (np.sqrt((X2_4RK[0] - X1_4RK[0])**2 + (Y2_4RK[0] - Y1_4RK[0])**2) * self.center.m)))

        EP2_4RK[0] = -((4 * np.pi**2 * self.satellite_2.m / np.sqrt(X2_4RK[0]**2 + Y2_4RK[0]**2)) + (
                4 * np.pi**2 * self.satellite_1.m * self.satellite_2.m / (np.sqrt((X2_4RK[0] - X1_4RK[0])**2 + (Y2_4RK[0] - Y1_4RK[0])**2) * self.center.m)))


        T_4RK[0] = 0

        for n in range(N - 1):
            k_x1 = self.t_step * U1_4RK[n]
            k_y1 = self.t_step * V1_4RK[n]
            k_u1 = self.t_step * HE(X1_4RK[n], Y1_4RK[n], X2_4RK[n], Y2_4RK[n])
            k_v1 = self.t_step * IE(X1_4RK[n], Y1_4RK[n], X2_4RK[n], Y2_4RK[n])

            l_x1 = self.t_step * U2_4RK[n]
            l_y1 = self.t_step * V2_4RK[n]
            l_u1 = self.t_step * HJ(X1_4RK[n], Y1_4RK[n], X2_4RK[n], Y2_4RK[n])
            l_v1 = self.t_step * IJ(X1_4RK[n], Y1_4RK[n], X2_4RK[n], Y2_4RK[n])

            k_x2 = self.t_step * (U1_4RK[n] + k_u1 / 2)
            k_y2 = self.t_step * (V1_4RK[n] + k_v1 / 2)
            k_u2 = self.t_step * HE(X1_4RK[n] + k_x1 / 2, Y1_4RK[n] + k_y1 / 2, X2_4RK[n] + l_x1 / 2, Y2_4RK[n] + l_y1 / 2)
            k_v2 = self.t_step * IE(X1_4RK[n] + k_x1 / 2, Y1_4RK[n] + k_y1 / 2, X2_4RK[n] + l_x1 / 2, Y2_4RK[n] + l_y1 / 2)

            l_x2 = self.t_step * (U2_4RK[n] + l_u1 / 2)
            l_y2 = self.t_step * (V2_4RK[n] + l_v1 / 2)
            l_u2 = self.t_step * HJ(X1_4RK[n] + k_x1 / 2, Y1_4RK[n] + k_y1 / 2, X2_4RK[n] + l_x1 / 2, Y2_4RK[n] + l_y1 / 2)
            l_v2 = self.t_step * IJ(X1_4RK[n] + k_x1 / 2, Y1_4RK[n] + k_y1 / 2, X2_4RK[n] + l_x1 / 2, Y2_4RK[n] + l_y1 / 2)

            k_x3 = self.t_step * (U1_4RK[n] + k_u2 / 2)
            k_y3 = self.t_step * (V1_4RK[n] + k_v2 / 2)
            k_u3 = self.t_step * HE(X1_4RK[n] + k_x2 / 2, Y1_4RK[n] + k_y2 / 2, X2_4RK[n] + l_x2 / 2, Y2_4RK[n] + l_y2 / 2)
            k_v3 = self.t_step * IE(X1_4RK[n] + k_x2 / 2, Y1_4RK[n] + k_y2 / 2, X2_4RK[n] + l_x2 / 2, Y2_4RK[n] + l_y2 / 2)

            l_x3 = self.t_step * (U2_4RK[n] + l_u2 / 2)
            l_y3 = self.t_step * (V2_4RK[n] + l_v2 / 2)
            l_u3 = self.t_step * HJ(X1_4RK[n] + k_x2 / 2, Y1_4RK[n] + k_y2 / 2, X2_4RK[n] + l_x2 / 2, Y2_4RK[n] + l_y2 / 2)
            l_v3 = self.t_step * IJ(X1_4RK[n] + k_x2 / 2, Y1_4RK[n] + k_y2 / 2, X2_4RK[n] + l_x2 / 2, Y2_4RK[n] + l_y2 / 2)

            k_x4 = self.t_step * (U1_4RK[n] + k_u3)
            k_y4 = self.t_step * (V1_4RK[n] + k_v3)
            k_u4 = self.t_step * HE(X1_4RK[n] + k_x3, Y1_4RK[n] + k_y3, X2_4RK[n] + l_x3, Y2_4RK[n] + l_y3)
            k_v4 = self.t_step * IE(X1_4RK[n] + k_x3, Y1_4RK[n] + k_y3, X2_4RK[n] + l_x3, Y2_4RK[n] + l_y3)

            l_x4 = self.t_step * (U2_4RK[n] + l_u3)
            l_y4 = self.t_step * (V2_4RK[n] + l_v3)
            l_u4 = self.t_step * HJ(X1_4RK[n] + k_x3, Y1_4RK[n] + k_y3, X2_4RK[n] + l_x3, Y2_4RK[n] + l_y3)
            l_v4 = self.t_step * IJ(X1_4RK[n] + k_x3, Y1_4RK[n] + k_y3, X2_4RK[n] + l_x3, Y2_4RK[n] + l_y3)

            X1_4RK[n + 1] = X1_4RK[n] + k_x1 / 6 + k_x2 / 3 + k_x3 / 3 + k_x4 / 6
            Y1_4RK[n + 1] = Y1_4RK[n] + k_y1 / 6 + k_y2 / 3 + k_y3 / 3 + k_y4 / 6
            U1_4RK[n + 1] = U1_4RK[n] + k_u1 / 6 + k_u2 / 3 + k_u3 / 3 + k_u4 / 6
            V1_4RK[n + 1] = V1_4RK[n] + k_v1 / 6 + k_v2 / 3 + k_v3 / 3 + k_v4 / 6

            X2_4RK[n + 1] = X2_4RK[n] + l_x1 / 6 + l_x2 / 3 + l_x3 / 3 + l_x4 / 6
            Y2_4RK[n + 1] = Y2_4RK[n] + l_y1 / 6 + l_y2 / 3 + l_y3 / 3 + l_y4 / 6
            U2_4RK[n + 1] = U2_4RK[n] + l_u1 / 6 + l_u2 / 3 + l_u3 / 3 + l_u4 / 6
            V2_4RK[n + 1] = V2_4RK[n] + l_v1 / 6 + l_v2 / 3 + l_v3 / 3 + l_v4 / 6

            EK1_4RK[n + 1] = 0.5 * self.satellite_1.m * (U1_4RK[n +1] ** 2 + V1_4RK[n +1] ** 2)
            EK2_4RK[n + 1] = 0.5 * self.satellite_2.m * (U2_4RK[n +1] ** 2 + V2_4RK[n +1] ** 2)

            EP1_4RK[n + 1] = -((4 * np.pi ** 2 * self.satellite_1.m / np.sqrt(X1_4RK[n + 1] ** 2 + Y1_4RK[n + 1] ** 2)) + (
                    4 * np.pi ** 2 * self.satellite_1.m * self.satellite_2.m / (
                    np.sqrt((X2_4RK[n + 1] - X1_4RK[n + 1]) ** 2 + (Y2_4RK[n + 1] - Y1_4RK[n + 1]) ** 2) * self.center.m)))

            EP2_4RK[n + 1] = -((4 * np.pi ** 2 * self.satellite_2.m / np.sqrt(X2_4RK[n + 1] ** 2 + Y2_4RK[n + 1] ** 2)) + (
                    4 * np.pi ** 2 * self.satellite_1.m * self.satellite_2.m / (
                    np.sqrt((X2_4RK[n + 1] - X1_4RK[n + 1]) ** 2 + (Y2_4RK[n + 1] - Y1_4RK[n + 1]) ** 2) * self.center.m)))

            T_4RK[n + 1] = (n + 1) * self.t_step


        return U1_4RK, V1_4RK, U2_4RK, V2_4RK, X1_4RK, Y1_4RK, X2_4RK, Y2_4RK, EK1_4RK, EK2_4RK, EP1_4RK, EP2_4RK, T_4RK