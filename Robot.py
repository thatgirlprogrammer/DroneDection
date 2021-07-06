from math import *
import random


class Robot:
    def __init__(self, length=20.0):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.length = length
        self.bearing_noise = 0.0
        self.steering_noise = 0.0
        self.distance_noise = 0.0

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y),
                                                str(self.orientation))

    def set(self, new_x, new_y, new_orientation):
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        self.bearing_noise = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    def measurement_prob(self, measurements):
        predicted_measurements = self.sense()  # Our sense function took 0 as an argument to switch off noise.
        error = 1.0
        for i in range(len(measurements)):
            error_bearing = abs(measurements[i] - predicted_measurements[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi  # truncate
            error *= (exp(- (error_bearing ** 2) / (self.bearing_noise ** 2) / 2.0) /
                      sqrt(2.0 * pi * (self.bearing_noise ** 2)))
        return error

    def move(self, motion):
        alphainit = motion[0]
        alpha = random.gauss(alphainit, steering_noise) % (2 * pi)
        d = motion[1] + random.gauss(0, distance_noise)
        if alphainit > max_steering_angle:
            raise ValueError('Car cannot turn with angle greater than pi/4')
        while alpha > max_steering_angle:
            alpha = random.gauss(alphainit, steering_noise) % (2 * pi)
        if alpha == 0.0:
            beta = 0.0
        else:
            beta = (d / self.length) * tan(alpha)
            R = d / beta

        if abs(beta) < 0.001:
            # straight movement
            x = self.x + d * cos(self.orientation)
            y = self.y + d * sin(self.orientation)
            thetanew = (self.orientation + beta) % (2 * pi)
        else:
            CX = self.x - sin(self.orientation) * R
            CY = self.y + cos(self.orientation) * R
            x = CX + sin(self.orientation + beta) * R
            y = CY - cos(self.orientation + beta) * R
            thetanew = (self.orientation + beta) % (2 * pi)

        result = Robot(self.length)
        result.set(x, y, thetanew)
        result.set_noise(bearing_noise, steering_noise, distance_noise)
        return result  # make sure your move function returns an instance

    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dx = landmarks[i][0] - self.x
            dy = landmarks[i][1] - self.y
            bearing = atan2(dy, dx) - self.orientation
            if dy < 0:
                bearing += 2 * pi
            Z.append(bearing)
        return Z


"""def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi))
                        + p[0].orientation - pi)
    return [x / len(p), y / len(p), orientation / len(p)]
"""

"""def generate_ground_truth(motions):
    myrobot = Robot()
    myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

    Z = []
    T = len(motions)

    for t in range(T):
        myrobot = myrobot.move(motions[t])
        Z.append(myrobot.sense())
    print('Robot:    ', myrobot)
    return [myrobot, Z]
"""

"""def print_measurements(Z):
    T = len(Z)

    print('measurements = [[%.8s, %.8s, %.8s, %.8s],' % \
          (str(Z[0][0]), str(Z[0][1]), str(Z[0][2]), str(Z[0][3])))
    for t in range(1, T - 1):
        print('                [%.8s, %.8s, %.8s, %.8s],' % (str(Z[t][0]), str(Z[t][1]), str(Z[t][2]), str(Z[t][3])))
    print('                [%.8s, %.8s, %.8s, %.8s]]' % (
        str(Z[T - 1][0]), str(Z[T - 1][1]), str(Z[T - 1][2]), str(Z[T - 1][3])))
"""

"""def check_output(final_robot1, estimated_position):
    error_x = abs(final_robot1.x - estimated_position[0])
    error_y = abs(final_robot1.y - estimated_position[1])
    error_orientation = abs(final_robot.orientation - estimated_position[2])
    error_orientation = (error_orientation + pi) % (2.0 * pi) - pi
    correct = error_x < tolerance_xy and error_y < tolerance_xy and error_orientation < tolerance_orientation
    return correct
"""

"""def particle_filter(motions1, measurements, N=500):  # I know it's tempting, but don't change N!
    p = []
    for i in range(N):
        r = Robot()
        r.set_noise(bearing_noise, steering_noise, distance_noise)
        p.append(r)

    for t in range(len(motions1)):
        p2 = []
        for i in range(N):
            p2.append(p[i].move(motions1[t]))
        p = p2

        w = []
        for i in range(N):
            w.append(p[i].measurement_prob(measurements[t]))

        # resampling
        p3 = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            p3.append(p[index])
        p = p3

    return get_position(p)
"""

# TEST CASES:
max_steering_angle = pi / 4.0
bearing_noise = 0.1
steering_noise = 0.1
distance_noise = 5.0
length = 20.0

tolerance_xy = 15.0
tolerance_orientation = 0.25
landmarks = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]  # position of 4 landmarks in (y, x) format.
world_size = 100.0  # world is NOT cyclic. Robot is allowed to travel "out of bounds"

myrobot = Robot(length)
myrobot.set(76.0, 94.0, 2.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

print('Robot:        ', myrobot)
print('Bearing measurements: ', myrobot.sense())

myrobot1 = Robot(length)
myrobot1.set(99.0, 36.0, pi / 7.0)
myrobot1.set_noise(bearing_noise, steering_noise, distance_noise)

print('Robot:        ', myrobot1)
print('Bearing measurements: ', myrobot1.sense())
