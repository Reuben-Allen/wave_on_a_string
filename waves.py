
# load python libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# declare oscillator class
class Oscillator:
    def __init__(self,length,k,resolution,mass_1,mass_2,damping=0):
        self.length = length
        self.k = k
        self.resolution = resolution
        self.mass = np.zeros(resolution)
        self.mass[0:int(np.round(resolution/2))] = mass_1
        self.mass[int(np.round(resolution/2)):] = mass_2
        self.x = np.linspace(0,self.length,resolution)
        self.damping = damping
    
    def oscillate(self,frequency,amplitude,time,fps,interference_frequency,interference_amplitude,pulse):
        # define array to hold wave information throughout the time interval
        # columns are individual masses and rows are time points
        time = round(time)
        fps = round(fps)
        body = np.zeros((time*fps,self.resolution))
        vel = np.zeros((time*fps,self.resolution))

        # define iteration parameters
        num_iter = time * fps * 100
        time_iter = time / num_iter
        body_iter = np.zeros(self.resolution)
        acc_iter = np.zeros(self.resolution-2)
        diff_iter = np.empty(self.resolution-1)
        vel_iter = np.zeros(self.resolution-2)

        # perform simulation
        if pulse:
            if interference_amplitude == 0 or interference_frequency == 0:
                # get half period
                period = 1/(frequency*2)

                # simulate motion of oscillator with a fixed end
                for i in range(num_iter):
                    time = i * time_iter
                    if time < period:
                        body_iter[0] = amplitude * np.sin(2*np.pi*frequency*time)
                    else:
                        body_iter[0] = 0
                    diff_iter = np.diff(body_iter)
                    acc_iter = np.divide(np.subtract(np.multiply(np.multiply(np.subtract(diff_iter[0:-1],diff_iter[1:]),self.k),-1),np.multiply(vel_iter,self.damping)),self.mass[1:-1])
                    vel_iter = np.add(vel_iter,np.multiply(acc_iter,time_iter))
                    body_iter[1:-1] = np.add(body_iter[1:-1],np.multiply(vel_iter,time_iter))
                    if i % 100 == 0:
                        body[int(i/100),:] = body_iter
                        vel[int(i/100),1:-1] = vel_iter
                    else:
                        continue
            else:
                # get half periods
                period_main = 1 / (frequency * 2)
                period_interference = 1 / (interference_frequency * 2)

                # simulate motion of oscillator with a driven end
                for i in range(num_iter):
                    time = i * time_iter
                    if time < period_main:
                        body_iter[0] = amplitude * np.sin(2*np.pi*frequency*time)
                    else:
                        body_iter[0] = 0
                    if time < period_interference:
                        body_iter[-1] = amplitude * np.sin(2*np.pi*interference_frequency*time)
                    else:
                        body_iter[-1] = 0
                    diff_iter = np.diff(body_iter)
                    acc_iter = np.divide(np.subtract(np.multiply(np.multiply(np.subtract(diff_iter[0:-1],diff_iter[1:]),self.k),-1),np.multiply(vel_iter,self.damping)),self.mass[1:-1])
                    vel_iter = np.add(vel_iter,np.multiply(acc_iter,time_iter))
                    body_iter[1:-1] = np.add(body_iter[1:-1],np.multiply(vel_iter,time_iter))
                    if i % 100 == 0:
                        body[int(i/100),:] = body_iter
                        vel[int(i/100),1:-1] = vel_iter
                    else:
                        continue
        else:
            if interference_amplitude == 0 or interference_frequency == 0:
                # simulate motion of oscillator with a fixed end
                for i in range(num_iter):
                    time = i * time_iter
                    body_iter[0] = amplitude * np.sin(2*np.pi*frequency*time)
                    diff_iter = np.diff(body_iter)
                    acc_iter = np.divide(np.subtract(np.multiply(np.multiply(np.subtract(diff_iter[0:-1],diff_iter[1:]),self.k),-1),np.multiply(vel_iter,self.damping)),self.mass[1:-1])
                    vel_iter = np.add(vel_iter,np.multiply(acc_iter,time_iter))
                    body_iter[1:-1] = np.add(body_iter[1:-1],np.multiply(vel_iter,time_iter))
                    if i % 100 == 0:
                        body[int(i/100),:] = body_iter
                        vel[int(i/100),1:-1] = vel_iter
                    else:
                        continue
            else:
                # simulate motion of oscillator with a driven end
                for i in range(num_iter):
                    time = i * time_iter
                    body_iter[0] = amplitude * np.sin(2*np.pi*frequency*time)
                    body_iter[-1] = interference_amplitude * np.sin(2*np.pi*interference_frequency*time)
                    diff_iter = np.diff(body_iter)
                    acc_iter = np.divide(np.subtract(np.multiply(np.multiply(np.subtract(diff_iter[0:-1],diff_iter[1:]),self.k),-1),np.multiply(vel_iter,self.damping)),self.mass[1:-1])
                    vel_iter = np.add(vel_iter,np.multiply(acc_iter,time_iter))
                    body_iter[1:-1] = np.add(body_iter[1:-1],np.multiply(vel_iter,time_iter))
                    if i % 100 == 0:
                        body[int(i/100),:] = body_iter
                        vel[int(i/100),1:-1] = vel_iter
                    else:
                        continue
        return body, vel

    def energy(self,body,vel,time):
        # returns a graph of the total energy of the oscillator as a function of time
        # body and vel are the outputs of the oscillate method and time is the simulated time interval
        # generate time values
        x = np.linspace(0,time,body.shape[0])

        # declare an array for storing energy values
        total_energy = np.zeros(body.shape[0])

        # calculate the total potential energy in the springs at each time point
        potential_energy = np.sum(np.divide(np.multiply(np.power(np.diff(body),2),self.k),2),axis=1)

        # calculate the total kinetic energy in the masses at each time point
        kinetic_energy = np.sum(np.divide(np.multiply(np.power(vel,2),self.mass),2),axis=1)

        # calculate the total mechanical energy
        total_energy = np.add(potential_energy,kinetic_energy)

        return total_energy, potential_energy, kinetic_energy, x

    def animate(self,frequency,amplitude,time=10,fps=30,interference_frequency=0,interference_amplitude=0,pulse=False,plot_energy=False):
        # get oscillator simulation
        body, vel = self.oscillate(frequency,amplitude,time,fps,interference_frequency,interference_amplitude,pulse)

        if plot_energy:
            # get energetic calculations
            total_energy, potential_energy, kinetic_energy, x = self.energy(body,vel,time)

            # initialize figure for plotting
            fig,(ax1,ax2) = plt.subplots(2,1)

            # setting plot limits and other aesthetics
            max_amplitude = np.max(body)
            max_energy = np.max(total_energy)
            ax1.set_xlim(0, self.length)
            ax1.set_ylim(-1*max_amplitude-max_amplitude/2, max_amplitude+max_amplitude/2)
            ax1.set_xlabel("Distance (m)")
            ax1.set_ylabel("Height (m)")
            ax2.set_xlim(0, time)
            ax2.set_ylim(0, max_energy+max_energy/2)
            ax2.set_ylabel("Total Energy (J)")
            ax2.set_xlabel("Time (s)")

            # plot energy
            ax2.plot(x,total_energy,color='k',label='Total Energy')
            ax2.plot(x,potential_energy,color='b',label='Potential Energy')
            ax2.plot(x,kinetic_energy,color='r',label='Kinetic Energy')
            ax2.legend(loc='best')

            # animate oscillator
            # initializing a line variable
            line, = ax1.plot(self.x, body[0,:])
            # update plot
            def frame(i):
                y_data = body[i,:]
                line.set_ydata(y_data)
                return line,

            animation = FuncAnimation(fig, func = frame, frames = np.arange(1,time*fps-1),interval=50)
            animation.save('wave_interference.gif')
            plt.show()

        else:
            # initialize figure for plotting
            fig,ax = plt.subplots()

            # setting plot limits and other aesthetics
            max = np.max(body)
            ax.set_xlim(0, self.length)
            ax.set_ylim(-1*max-max/2, max+max/2)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Height (m)")
    
            # initializing a line variable
            line, = ax.plot(self.x, body[0,:])
            # update plot
            def frame(i):
                y_data = body[i,:]
                line.set_ydata(y_data)
                return line,

            animation = FuncAnimation(fig, func = frame, frames = np.arange(1,time*fps-1),interval=50)
            #animation.save('wave_interference.gif')
            plt.show()
        return

if __name__ == "__main__":
    spring = Oscillator(10,20,1000,0.05,0.05,0)
    spring.animate(0.05,0.9,time=150,fps=1,interference_frequency=0.1,interference_amplitude=0.3,pulse=False,plot_energy=False)