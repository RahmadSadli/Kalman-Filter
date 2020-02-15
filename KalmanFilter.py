import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc

        self.A = np.matrix([[1, self.dt],
                            [0, 1]])
        self.B = np.matrix([[(self.dt**2)/2], [self.dt]]) * self.std_acc**2

        self.H = np.matrix([[1,0]])

        self.Q = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                            [(self.dt**3)/2, self.dt**2]])
        self.R = std_meas**2

        self.P = np.eye(self.A.shape[1])

        #self.x = np.matrix([[np.random.uniform(-200, 15)], [0]])
        self.x = np.matrix([[0],[0]])

    def predict(self):
        # Ref :Eq.(9) and Eq.(10)

        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Ex
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)

        # S = H*P*H'+Ez
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+Ez)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)


def main():

    #print(x)
    dt=0.1
    #x = np.linspace(0, 1, (1//dt)+1)
    x = np.arange(0, 20, dt)

    # Define a model track
    real_track = 0.25*(x**3 - 10*(x**2)) #+ 10*x + 15
    #real =(x+3)*((x-2)**2)*((x + 1)**3)


    u=1.5
    std_acc = 0.005
    std_meas = 0.5
    noise_magnitude = 25

    # create KalmanFilter object
    kf = KalmanFilter(dt, u, std_acc, std_meas)

    predictions = []
    measurements=[]
    for z in real_track:
        #Mesurement
        z=kf.H*z+ std_meas*np.random.normal(0, noise_magnitude, 1)

        measurements.append(z.item(0))
        predictions.append(kf.predict()[0])
        kf.update(z.item(0))

    fig = plt.figure()

    fig.suptitle('Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)

    plt.plot(x, measurements, label='Measurements', color='b')

    plt.plot(x, np.squeeze(predictions), label='Kalman Filter Prediction', color='r')

    plt.plot(x, np.array(real_track), label='Real Track', color='k')
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
