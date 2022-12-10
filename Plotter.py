import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import matplotlib.colors as colors


class Plotter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", 34567))
        self.fig = plt.figure(figsize=(15, 15))
        ax0 = plt.subplot(2, 3, 1)
        plt.title("Gaussian Receptive Fields")
        plt.ylabel("Neuron ID")
        plt.xlabel("Input dimensions")
        plt.xticks(
            [0, 1, 2, 3, 4, 5, 6, 7],
            labels=[
                "X pos",
                "Y pos",
                "X vel",
                "Y vel",
                "Ang",
                "A vel",
                "L leg",
                "R leg",
            ],
            fontsize="small",
            rotation=0,
        )
        self.gaussIn = plt.imshow(
            np.random.rand(8, 64).T,
            aspect=0.1,
            interpolation="nearest",
            norm=colors.SymLogNorm(
                linthresh=0.0003, linscale=0.003, vmin=0.0, vmax=1.0, base=10
            ),
        )
        ax0 = plt.subplot(2, 3, 2)
        plt.title("1st Layer Spikes")
        plt.xlabel("Simulation time")
        plt.ylabel("Neuron ID")
        self.line0 = plt.imshow(
            np.random.rand(16, 128).T, aspect=0.3, interpolation="nearest"
        )
        ax1 = plt.subplot(2, 3, 3)
        # plt.ylim([-5,5])
        plt.title("1st Layer Membrane Potentials")
        plt.xlabel("Simulation time")
        plt.ylabel("Neuron ID")
        # self.plotlines1 = plt.plot(np.arange(16), np.random.rand(16, 128))
        self.line1 = plt.imshow(
            np.random.rand(16, 128).T,
            aspect=0.1,
            interpolation="nearest",
            norm=colors.Normalize(vmin=0.0, vmax=3.0),
        )
        ax2 = plt.subplot(2, 3, 5)
        plt.ylim([0, 1.2])
        plt.title("Normalized Q-value differences")
        plt.ylabel("L1 Norm. Q-val. diff. from min")
        plt.xlabel("Actions")
        self.bar = plt.bar(
            ["Do nothing", "Right engine", "Main engine", "Left engine"], [0, 0, 0, 0]
        )

        ax3 = plt.subplot(2, 3, 4)
        plt.ylim([0, 180])
        plt.title("Readout neuron membrane potential")
        plt.ylabel("Membrane potential")
        plt.xlabel("Simulation time")

        self.plotlines2 = plt.plot(np.arange(16), np.random.rand(16, 4))
        plt.legend(
            ["Do nothing", "Right engine", "Main engine", "Left engine"],
            loc="upper left",
        )
        ax5 = plt.subplot(2, 3, 6)
        plt.title("Simulation")
        self.envrender = plt.imshow(np.random.rand(400, 600, 3))

    def update(self, *args):
        try:
            datarec, gaussIn, img = pickle.loads(self.sock.recv(10000000))
            self.line0.set_array(datarec[0][:, 0, :].T)
            self.line1.set_array(datarec[-3][:, 0, :].T)
            normedQ = datarec[-1][-1, :, :].copy()
            normedQ -= np.min(normedQ)
            normedQ = normedQ / np.sum(normedQ)
            [self.bar[i].set_height(normedQ[0, i]) for i in range(4)]
            [self.plotlines2[i].set_ydata(datarec[-1][:, 0, i]) for i in range(4)]
            # [self.plotlines1[i].set_ydata(datarec[-3][:, 0, i]) for i in range(128)]
            self.gaussIn.set_array(gaussIn.T)
            self.envrender.set_array(img)
        except:
            pass
        return [
            self.line0,
            self.line1,
            *self.bar,
            *self.plotlines2,
            self.gaussIn,
            self.envrender,
        ]  # , *self.plotlines1]


if __name__ == "__main__":
    plot = Plotter()
    animation = FuncAnimation(plot.fig, plot.update, frames=100, interval=20, blit=True)
    plt.show()
