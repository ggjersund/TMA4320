# Dependencies
import numpy as np
import pickle
import random

# Function used to plot gravity for exercise 1
def plotgravity(xvalue, distance, y1, y2):
    return (((y2 - xvalue) / (distance * (((distance ** 2) + ((xvalue - y2) ** 2)) ** (1 / 2)))) - (
        (y1 - xvalue) / (distance * (((distance ** 2) + ((xvalue - y1) ** 2)) ** (1 / 2)))))

# Create Chebyshev interpolation points
def chebyshev(a, b, N):
    I = np.arange(1, N + 1, 1)
    X = (b + a) / 2 + (b - a) / 2 * np.cos((2 * I - 1) * np.pi / (2 * N))

    return X

# Create a midpoint quadrature
def midpoint(a, b, N):
    uniform = np.linspace(a, b, N + 1)

    weights = np.diff(uniform)
    points = (weights / 2) + (uniform[:-1])

    return points, weights

# Create a Legendre-Gauss quadrature
def legendre(a, b, N):
    points, weights = np.polynomial.legendre.leggauss(N)
    points = ((points / 2) * (b - a)) + ((b + a) / 2)
    weights = (b - a) * (weights / 2)

    return points, weights

# Calculate the j'th Lagrange basis polynomial
def lagrange(xqk, xs, j):
    lag = lambda y: np.prod((y - xs[np.arange(xs.shape[0]) != j]))
    return (lag(xqk) / lag(xs[j]))

# Calculate the "analytic" rho
def rho(omega, gamma, xs):
    rho = np.sin(omega * xs) * np.exp(gamma * xs)
    return rho

# The provided function kernel
def kernel(d, xci, xqk):
    return (d / ((((d ** 2) + ((xci - xqk) ** 2)) ** (3 / 2))))

# Calculate the left-hand side of the equation Ax = b
def fredholm_lhs(d, xc, xs, xq, w, K):

    Nc = xc.shape[0]
    Ns = xs.shape[0]
    Nq = xq.shape[0]

    A = np.zeros((Nc, Ns))

    lagrange_vector = np.zeros((Ns, Nq))

    for j in range(Ns):
        for k in range(Nq):
            lagrange_vector[j][k] = lagrange(xq[k], xs, j)

    for i in range(Nc):
        for j in range(Ns):
            summarize = 0
            for k in range(Nq):
                summarize += w[k] * K(d, xc[i], xq[k]) * lagrange_vector[j][k]

            A[i, j] = summarize

    return A

# Calculate the right-hand side of the equation Ax = b
# This is "analytic" and based on provided code
def fredholm_rhs(a, b, omega, gamma, d, x_eval):
    from test_example import analytical_solution

    try:
        F = pickle.load(open("F.pkl", "rb"))

    except:
        F = analytical_solution(a, b, omega, gamma, 75)
        pickle.dump(F, open("F.pkl", "wb"))

    F(x_eval, d)

    return F(x_eval, d)

# Return a epsilon vector
def epsilon(xc):
    n = xc.shape[0]
    epsilon_vector = np.zeros(n)

    for i in range(n):
        epsilon_vector[i] = random.uniform(-10**(-3), 10**(-3))

    return epsilon_vector

# Create a logarithmic equally seperated vector of Lambda values
def lambda_list(N):
    return np.geomspace(10**(-14), 10, N)
