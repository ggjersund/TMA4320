import numpy as np
import random
import decimal

decimal.getcontext().prec = 200


class Protein:

    # Initialize variables
    def __init__(self, n, t):

        self.count = 0
        self.T = t
        self.length = n
        self.matrix = np.zeros((self.length, 2))
        self.U_matrix = (6.93 * 10 ** (-21)) * np.random.random([self.length, self.length]) - 10.4 * 10 ** (-21)
        self.E = 0

        for i in range(self.length):
            self.matrix[i][0] = i
            self.matrix[i][1] = self.length // 2

    # Return the point to turn about
    def get_twist_point(self):
        return random.randint(1, self.length - 1)

    # Check if twist is valid
    def invalid_twist(self, twist_point, x, y):
        if twist_point > ((self.length / 2) - 1):

            # Check against all points less than the twist point (excluded)
            for i in range(twist_point - 1, -1, -1):
                if (self.matrix[i][0] == x) and (self.matrix[i][1] == y):
                    return True
        else:

            # Check against all points more than the twist point (excluded)
            for i in range(twist_point + 1, self.length, 1):
                if (self.matrix[i][0] == x) and (self.matrix[i][1] == y):
                    return True

        # Not collision found -> valid twist
        return False

    # Perform one twist
    def twist(self):
            matrix_copy = np.copy(self.matrix)

            # Loop until legal twist is found
            while True:

                illegal = False
                twist_point = self.get_twist_point()
                choose_flip = bool(random.getrandbits(1))

                # Find what end to twist
                if twist_point > ((self.length / 2) - 1):

                    # Count from twist point (excluded) and up
                    for j in range(twist_point + 1, self.length, 1):

                        # Perform random twist
                        if choose_flip:
                            x, y = self.twist_left(matrix_copy[twist_point], matrix_copy[j])
                        else:
                            x, y = self.twist_right(matrix_copy[twist_point], matrix_copy[j])

                        # Check if twist is invalid
                        if self.invalid_twist(twist_point, x, y):
                            matrix_copy = np.copy(self.matrix)
                            illegal = True
                            break
                        else:
                            matrix_copy[j] = np.array([x, y])

                else:

                    # Count from twist point (excluded) and down
                    for j in range(twist_point - 1, -1, -1):

                        # Perform random twist
                        if choose_flip:
                            x, y = self.twist_left(matrix_copy[twist_point], matrix_copy[j])
                        else:
                            x, y = self.twist_right(matrix_copy[twist_point], matrix_copy[j])

                        # Check if twist is invalid
                        if self.invalid_twist(twist_point, x, y):
                            matrix_copy = np.copy(self.matrix)
                            illegal = True
                            break
                        else:
                            matrix_copy[j] = np.array([x, y])

                # Check energy and random fluctuations
                if not illegal:

                    # Check matrix energy
                    energy = self.check_energy(matrix_copy)

                    # Check if lower than existing energy level
                    if energy <= self.E:
                        self.matrix = np.copy(matrix_copy)
                        self.E = energy

                    # Check for thermal fluctuation
                    elif self.T > 1:
                        if random.uniform(0, 1) < decimal.Decimal((self.E - energy) / (self.T * 1.38064853*(10**(-23)))).exp():
                            self.matrix = np.copy(matrix_copy)
                            self.E = energy

                    # Break out of while statement
                    break

    # Check matrix energy
    def check_energy(self, matrix):
        energy = 0

        # Loop through all the points and check nearest neighbor
        for i in range(0, self.length):
            # Loop up from i element by offset of +3
            for j in range(i + 3, self.length):
                if np.array_equal(matrix[i] - matrix[j], [0., 1.]) or \
                        np.array_equal(matrix[i] - matrix[j], [1., 0.]):
                    energy += self.U_matrix[i][j]

            # Loop down from element i by offset of -3
            for k in range(i - 3, -1, -1):
                if np.array_equal(matrix[i] - matrix[k], [0., 1.]) or \
                        np.array_equal(matrix[i] - matrix[k], [1., 0.]):
                    energy += self.U_matrix[i][k]

        return energy

    # Return energy
    def energy(self):
        return self.E

    # Perform left twist
    @staticmethod
    def twist_left(start_point, current_point):
        """
        +x -> -y
        -x -> +y
        +y -> +x
        -y -> -x
        """
        x = int(start_point[0] + (current_point - start_point)[1])
        y = int(start_point[1] - (current_point - start_point)[0])

        return x, y

    # Perform right twist
    @staticmethod
    def twist_right(start_point, current_point):
        """
        +y -> -x
        -y -> +x
        +x -> +y
        -x -> -y
        """

        x = int(start_point[0] - (current_point - start_point)[1])
        y = int(start_point[1] + (current_point - start_point)[0])

        return x, y

    # Return diameter
    def diameter(self):
        max_distance = 0
        for i in self.matrix:
            for j in self.matrix:
                dist = ((abs(j[0] - i[0]) + 1) ** 2 + (abs(j[1] - i[1]) + 1) ** 2) ** (1/2)
                if dist > max_distance:
                    max_distance = dist
        return max_distance

    # Change temperature
    def change_temp(self, temp):
        self.T = temp


class Grid:
    """
    Create the grid to fill in
    """
    def __init__(self, N):
        self.matrix = np.zeros((N, N), dtype=int)
        self.length = N

    def fill(self, matrix):
        for i in range(self.length):
            x = int(matrix[i][0])
            q = int(matrix[i][1])
            self.matrix[x][q] = int((i + 1))