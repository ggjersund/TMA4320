# !!!! Requirements for mac !!!!
# !!!! comment out this on PC !!!!
import matplotlib
matplotlib.use('TkAgg')
# !!!! up to this point !!!!

# Dependencies
import matplotlib.pyplot as plt
import numpy as np
import time

# Import functions
from functions import chebyshev, midpoint, legendre, rho, kernel, fredholm_lhs, fredholm_rhs, plotgravity, epsilon, lambda_list


def exercise1():

    # Generate numpy array
    x = np.arange(0, 1, 0.01)

    # Constants
    y1 = 1 / 3
    y2 = 2 / 3

    # Depths
    d1 = 0.025
    d2 = 0.25
    d3 = 2.5

    plt.figure('exercise-1')
    plt.title('Gravitational force')
    plt.semilogy(x, plotgravity(x, d1, y1, y2), label='$d = 0.025$')
    plt.semilogy(x, plotgravity(x, d2, y1, y2), label='$d = 0.250$')
    plt.semilogy(x, plotgravity(x, d3, y1, y2), label='$d = 2.500$')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

def exercise3():

    # Orders
    Nc = 40  # Number of collocation points
    Ns = Nc  # Number of source points
    Nq = 400  # Number of panels

    # Limits
    a = 0
    b = 1

    # Constants
    omega = 3 * np.pi
    gamma = -2

    # Depths
    d = 0.025

    # Create points
    xc = chebyshev(a, b, Nc)  # Collocation points
    xs = chebyshev(a, b, Ns)  # Source points

    # Calculate rho vector
    rho_vector = rho(omega, gamma, xs)

    # Calculate A-matrix from midpoint quadrature
    xq, w = midpoint(a, b, Nq)
    a_matrix_midpoint = fredholm_lhs(d, xc, xs, xq, w, kernel)

    # Calculate numerical B-vector
    b_numeric_midpoint = np.dot(a_matrix_midpoint, rho_vector)

    # Calculate analytic B-vector
    b_analytic = fredholm_rhs(a, b, omega, gamma, d, xc)

    # Plot integration
    plt.figure("integration-plot")
    plt.title("Analytic and numerical integration")
    plt.plot(xc, b_analytic, label="Analytic")
    plt.plot(xc, b_numeric_midpoint, label="Midpoint")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)

    # Get error by # of panels
    error_midpoint = np.zeros(Nq - 2)
    error_axis = np.arange(1, Nq - 1, 1)

    for i in range(1, Nq - 1, 1):
        # Print each iteration
        print(i)

        # Calculate A-matrix from midpoint quadrature
        xq, w = midpoint(a, b, i)
        a_matrix_midpoint = fredholm_lhs(d, xc, xs, xq, w, kernel)

        # Calculate numerical B-vector
        b_numeric_midpoint = np.dot(a_matrix_midpoint, rho_vector)

        # Calculate error
        print(b_analytic.shape[0])
        error_midpoint[i - 1] = np.amax(np.absolute(np.subtract(b_analytic, b_numeric_midpoint)))

    plt.figure("error-plot")
    plt.title("Midpoint integration error")
    plt.semilogy(error_axis, error_midpoint, label="Midpoint error")
    plt.xlabel('Nq')
    plt.ylabel('Max integration error')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise4():

    # Orders
    Nc = 40  # Number of collocation points
    Ns = Nc  # Number of source points
    Nq = 400  # Number of panels

    # Limits
    a = 0
    b = 1

    # Constants
    omega = 3 * np.pi
    gamma = -2

    # Depths
    d = 0.025

    # Create points
    xc = chebyshev(a, b, Nc)  # Collocation points
    xs = chebyshev(a, b, Ns)  # Source points

    # Calculate rho vector
    rho_vector = rho(omega, gamma, xs)

    # Calculate A-matrix from Legendre-Gauss quadrature
    xq, w = legendre(a, b, Nq)

    a_matrix_legendre = fredholm_lhs(d, xc, xs, xq, w, kernel)

    # Calculate numerical B-vector
    b_numeric_legendre = np.dot(a_matrix_legendre, rho_vector)

    # Calculate analytic B-vector
    b_analytic = fredholm_rhs(a, b, omega, gamma, d, xc)

    # Plot integration
    plt.figure("integration-plot")
    plt.title("Analytic and numerical integration")
    plt.plot(xc, b_analytic, label="Analytic")
    plt.plot(xc, b_numeric_legendre, label="Legendre")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)

    # Get error by # of panels
    error_legendre = np.zeros(Nq - 2)
    error_axis = np.arange(1, Nq - 1, 1)

    for i in range(1, Nq - 1, 1):
        # Print each iteration
        print(i)

        # Calculate A-matrix from Legendre-Gauss quadrature
        xq, w = legendre(a, b, i)
        a_matrix_legendre = fredholm_lhs(d, xc, xs, xq, w, kernel)

        # Calculate numerical B-vector
        b_numeric_legendre = np.dot(a_matrix_legendre, rho_vector)

        # Calculate error
        error_legendre[i - 1] = np.amax(np.absolute(np.subtract(b_analytic, b_numeric_legendre)))

    plt.figure("error-plot")
    plt.title("Legendre-Gauss integration error")
    plt.semilogy(error_axis, error_legendre, label="Legendre error")
    plt.xlabel('Nq')
    plt.ylabel('Max integration error')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise5():

    # Orders
    Nc = 30         # Number of collocation points
    Nq = Nc**2      # Number of panels

    # Limits
    a = 0
    b = 1

    # Constants
    omega = 3 * np.pi
    gamma = -2

    # Depths
    d1 = 0.025
    d2 = 0.25
    d3 = 2.5

    # Error
    error1 = np.zeros(Nc - 5)
    error2 = np.zeros(Nc - 5)
    error3 = np.zeros(Nc - 5)
    axis = np.arange(5, Nc, 1)

    for i in range(5, Nc, 1):
        # Print each iteration
        print(i)

        # Create points
        xc = chebyshev(a, b, i)  # Collocation points
        xs = chebyshev(a, b, i)  # Source points

        # Calculate true rho
        rho_analytic = rho(omega, gamma, xs)

        # Calculate A-matrix from Legendre-Gauss quadrature
        xq, w = legendre(a, b, Nq)
        a_matrix_legendre1 = fredholm_lhs(d1, xc, xs, xq, w, kernel)
        a_matrix_legendre2 = fredholm_lhs(d2, xc, xs, xq, w, kernel)
        a_matrix_legendre3 = fredholm_lhs(d3, xc, xs, xq, w, kernel)

        # Find rho by linear algebra
        rho_numeric1 = np.linalg.solve(a_matrix_legendre1, fredholm_rhs(a, b, omega, gamma, d1, xc))
        rho_numeric2 = np.linalg.solve(a_matrix_legendre2, fredholm_rhs(a, b, omega, gamma, d2, xc))
        rho_numeric3 = np.linalg.solve(a_matrix_legendre3, fredholm_rhs(a, b, omega, gamma, d3, xc))

        error1[i - 5] = np.amax(np.absolute(np.subtract(rho_numeric1, rho_analytic)))
        error2[i - 5] = np.amax(np.absolute(np.subtract(rho_numeric2, rho_analytic)))
        error3[i - 5] = np.amax(np.absolute(np.subtract(rho_numeric3, rho_analytic)))

    plt.figure("inverse-error")
    plt.title("Numerical solution for Rho")
    plt.semilogy(axis, error1, label="$d = 0.025$")
    plt.semilogy(axis, error2, label="$d = 0.25$")
    plt.semilogy(axis, error3, label="$d = 2.5$")
    plt.xlabel('Nc')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise6():

    # Orders
    Nc = 30         # Number of collocation points
    Nq = Nc**2      # Number of panels

    # Limits
    a = 0
    b = 1

    # Constants
    omega = 3 * np.pi
    gamma = -2

    # Depths
    d1 = 0.025
    d2 = 0.25
    d3 = 2.5

    # Create chebyshev points
    xc = chebyshev(a, b, Nc)  # Collocation points
    xs = chebyshev(a, b, Nc)  # Source points

    """
    # Plot 1 (d = 0.025)
    b_analytic1 = fredholm_rhs(a, b, omega, gamma, d1, xc)
    b_analytic_disturbed1 = fredholm_rhs(a, b, omega, gamma, d1, xc) * (1 + epsilon(xc))
    
    plt.figure("inverse-error")
    plt.title("Analytic with and without random noise")
    plt.plot(xc, b_analytic1, label="Analytic, $d = 0.025$")
    plt.plot(xc, b_analytic_disturbed1, label="Analytic disturbed, $d = 0.025$")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2 (d = 0.25)
    b_analytic2 = fredholm_rhs(a, b, omega, gamma, d2, xc)
    b_analytic_disturbed2 = fredholm_rhs(a, b, omega, gamma, d2, xc) * (1 + epsilon(xc))

    plt.figure("inverse-error")
    plt.title("Analytic with and without random noise")
    plt.plot(xc, b_analytic2, label="Analytic, $d = 0.25$")
    plt.plot(xc, b_analytic_disturbed2, label="Analytic disturbed, $d = 0.25$")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 3 (d = 2.5)
    b_analytic3 = fredholm_rhs(a, b, omega, gamma, d3, xc)
    b_analytic_disturbed3 = fredholm_rhs(a, b, omega, gamma, d3, xc) * (1 + epsilon(xc))

    plt.figure("inverse-error")
    plt.title("Analytic with and without random noise")
    plt.plot(xc, b_analytic3, label="Analytic, $d = 2.5$")
    plt.plot(xc, b_analytic_disturbed3, label="Analytic disturbed, $d = 2.5$")
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    # Rho analytic
    rho_analytic = rho(omega, gamma, xs)
    xq, w = legendre(a, b, Nq)

    # Plot 4
    a_matrix_legendre1 = fredholm_lhs(d1, xc, xs, xq, w, kernel)
    rho_numeric1 = np.linalg.solve(a_matrix_legendre1, fredholm_rhs(a, b, omega, gamma, d1, xc))
    rho_numeric_disturbed1 = np.linalg.solve(a_matrix_legendre1, fredholm_rhs(a, b, omega, gamma, d1, xc) * (1 + epsilon(xc)))

    plt.figure("inverse-error")
    plt.title("Rho")
    plt.plot(xc, rho_analytic, label="Analytic, $d = 0.025$")
    plt.plot(xc, rho_numeric1, label="Numeric, $d = 0.025$")
    plt.plot(xc, rho_numeric_disturbed1, label="Numeric disturbed, $d = 0.025$")
    plt.xlabel('x')
    plt.ylabel('Rho(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 5
    a_matrix_legendre2 = fredholm_lhs(d2, xc, xs, xq, w, kernel)
    rho_numeric2 = np.linalg.solve(a_matrix_legendre2, fredholm_rhs(a, b, omega, gamma, d2, xc))
    rho_numeric_disturbed2 = np.linalg.solve(a_matrix_legendre2, fredholm_rhs(a, b, omega, gamma, d2, xc) * (1 + epsilon(xc)))

    plt.figure("inverse-error")
    plt.title("Rho")
    plt.plot(xc, rho_analytic, label="Analytic, $d = 0.25$")
    plt.plot(xc, rho_numeric2, label="Numeric, $d = 0.25$")
    plt.plot(xc, rho_numeric_disturbed2, label="Numeric disturbed, $d = 0.25$")
    plt.yscale('symlog')
    plt.xlabel('x')
    plt.ylabel('Rho(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 6
    a_matrix_legendre3 = fredholm_lhs(d3, xc, xs, xq, w, kernel)
    rho_numeric3 = np.linalg.solve(a_matrix_legendre3, fredholm_rhs(a, b, omega, gamma, d3, xc))
    rho_numeric_disturbed3 = np.linalg.solve(a_matrix_legendre3, fredholm_rhs(a, b, omega, gamma, d3, xc) * (1 + epsilon(xc)))

    plt.figure("inverse-error")
    plt.title("Rho")
    plt.plot(xc, rho_analytic, label="Analytic, $d = 2.5$")
    plt.plot(xc, rho_numeric3, label="Numeric, $d = 2.5$")
    plt.plot(xc, rho_numeric_disturbed3, label="Numeric disturbed, $d = 2.5$")
    plt.yscale('symlog')
    plt.xlabel('x')
    plt.ylabel('Rho(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise7():
    # Orders
    Nc = 30  # Number of collocation points
    Ns = 30
    Nq = Nc ** 2  # Number of panels

    # Limits
    a = 0
    b = 1

    # Constants
    omega = 3 * np.pi
    gamma = -2

    # Depths
    d2 = 0.25
    d3 = 2.5

    # Create chebyshev points
    xc = chebyshev(a, b, Nc)  # Collocation points
    xs = chebyshev(a, b, Nc)  # Source points

    rho_analytic = rho(omega, gamma, xs)

    # Analytic disturbed b-vector
    b_analytic_disturbed2 = fredholm_rhs(a, b, omega, gamma, d2, xc) * (1 + epsilon(xc))
    b_analytic_disturbed3 = fredholm_rhs(a, b, omega, gamma, d3, xc) * (1 + epsilon(xc))

    xq, w = legendre(a, b, Nq)
    lambda_vector = lambda_list(16)
    identity_matrix = np.identity(Nc)
    rho_vector2 = np.zeros(lambda_vector.shape[0])
    rho_vector3 = np.zeros(lambda_vector.shape[0])

    # Calculate A-matrix from Legendre-Gauss quadrature
    a_matrix2 = fredholm_lhs(d2, xc, xs, xq, w, kernel)
    a_matrix_transposed2 = np.transpose(a_matrix2)
    a_matrix3 = fredholm_lhs(d3, xc, xs, xq, w, kernel)
    a_matrix_transposed3 = np.transpose(a_matrix3)


    for j in range(lambda_vector.shape[0]):

            # Calculate 2
            left_side = np.matmul(a_matrix_transposed2, a_matrix2) + (lambda_vector[j] * identity_matrix)
            right_side = np.matmul(a_matrix_transposed2, b_analytic_disturbed2)
            rho_vector2[j] = np.amax(np.absolute(np.subtract(np.linalg.solve(left_side, right_side), rho_analytic)))

            # Calculate 3
            left_side = np.matmul(a_matrix_transposed3, a_matrix3) + (lambda_vector[j] * identity_matrix)
            right_side = np.matmul(a_matrix_transposed3, b_analytic_disturbed3)
            rho_vector3[j] = np.amax(np.absolute(np.subtract(np.linalg.solve(left_side, right_side), rho_analytic)))


    plt.figure("tikhonov")
    plt.title("Rho error")
    plt.loglog(lambda_vector, rho_vector2, label="Numeric, $d = 0.25$")
    plt.loglog(lambda_vector, rho_vector3, label="Numeric, $d = 2.5$")
    plt.xlabel('Lambda')
    plt.ylabel('Rho(x) error')
    plt.legend()
    plt.grid(True)
    plt.show()

def exercise8():
    file_name = 'q8_3.npz'
    f = open(file_name, 'rb')
    npzfile = np.load(f)

    a = npzfile['a']
    b = npzfile['b']
    d = npzfile['d']
    xc = npzfile['xc']
    b_measured = npzfile['F']

    # Create source points
    n = xc.shape[0]
    xs = chebyshev(a, b, n)   # Source points
    lambda_select = 10**(-4)

    print(a)
    print(b)
    print(d)
    print(xc.shape[0])

    # Calculate A-matrix from Legendre-Gauss quadrature
    xq, w = legendre(a, b, 2 * n**2)
    a_matrix = fredholm_lhs(d, xc, xs, xq, w, kernel)

    identity_matrix = np.identity(a_matrix.shape[0])
    a_matrix_transposed = np.transpose(a_matrix)

    left_side = np.matmul(a_matrix_transposed, a_matrix) + (lambda_select * identity_matrix)
    right_side = np.matmul(a_matrix_transposed, b_measured)

    rho_solution = np.linalg.solve(left_side, right_side)

    plt.figure("measurement-8-3")
    plt.title("Solution to measurement 3")
    plt.plot(xc, rho_solution, label="Numeric, $\lambda = 10^{-4}$")
    plt.xlabel('xc')
    plt.ylabel('Rho(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Change the number in order to calculate a given exercise
# (Exercise 2 is provided in LaTeX)
exercise5()