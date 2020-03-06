import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from pylab import *


class FluidSolver(object):
    def __init__(self, flow, viscosity=0.005, force_function=None, advection_method="warp"):
        """
        Simple fluid solver based on:
        * Applying a time-dependent force to a velocity field (flow)
        * Advecting the flow by itself
        * Using FFT tricks to model viscosity and mass-conservation
        """
        dims = len(flow.shape) - 1
        if flow.shape[-1] != dims:
            raise ValueError("{0}-dimensional flow must have {0} components".format(dims))
        if dims != 2:
            raise NotImplementedError("Only 2D implemented")
        if flow.shape[0] != flow.shape[1]:
            raise NotImplementedError("Only square grids supported")
        self.flow = tf.expand_dims(tf.constant(flow), 0)
        self.viscosity = viscosity
        self.force_function = force_function
        self.t = 0

        def self_advection_warp(flow):
            """
            Fluid flow is modeled as a velocity field.

            Imagine a dust particle being dragged along the flow, this is advection.
            Self-advection is the flow dragging itself forward.
            """
            # XXX: tensorflow_addons seems to have a bug where the x and y flow components are swapped.
            return tfa.image.dense_image_warp(flow, flow[:,:,:,::-1])

        # Calculate wave numbers for FFT
        omega = np.arange(flow.shape[0])
        omega -= flow.shape[0] * (2*omega > flow.shape[0])
        omega_x, omega_y = np.meshgrid(omega, omega)
        decay = tf.constant(np.exp(-viscosity*(omega_x**2 + omega_y**2)), "complex128")
        r = np.sqrt(omega_x**2 + omega_y**2)
        unit_x = tf.constant(omega_x / (r + 1e10*(r==0)), "complex128")
        unit_y = tf.constant(omega_y / (r + 1e10*(r==0)), "complex128")
        def viscosity_and_mass_conservation(flow):
            u = flow[0,:,:,0]
            v = flow[0,:,:,1]
            waves_x = tf.signal.fft2d(tf.cast(u, "complex128"))
            waves_y = tf.signal.fft2d(tf.cast(v, "complex128"))
            # === Viscosity ===
            # Viscosity is modeled as the flow spreading out in the same way as heat spreads in the Heat Equation.
            # This is done by suppressing the high-frequency components with an exponential decay.
            waves_x *= decay
            waves_y *= decay
            # === Mass conservation ===
            # Getting rid of divergence is surprisingly easy in the Fourier domain.
            # Notice how in the space domain the complex plane wave (0, exp(1j*omega*x))
            # is divergence-free as the y-component only depends on the x-coordinate.
            # Any scaled/rotated version is also divergence-free.
            # The code bellow gets rid of all Fourier components that have a divergence.
            # If plotted this looks like a swirl around the origin Fourier-origin.
            source_field = waves_x * unit_x + waves_y * unit_y
            waves_x = waves_x - unit_x * source_field
            waves_y = waves_y - unit_y * source_field
            u = tf.cast(tf.signal.ifft2d(waves_x), "float64")
            v = tf.cast(tf.signal.ifft2d(waves_y), "float64")
            return tf.expand_dims(tf.stack([u, v], axis=-1), 0)

        # TODO: Fix scaling to produce a comparable effect to the warp method
        def self_advection_gradient(flow):
            """
            Self-advection implemented using the gradient method.
            """
            u = flow[0,:,:,0]
            v = flow[0,:,:,1]
            waves_x = tf.signal.fft2d(tf.cast(u, "complex128"))
            waves_y = tf.signal.fft2d(tf.cast(v, "complex128"))

            u_dx = tf.cast(tf.signal.ifft2d(waves_x * 1j * omega_x), "float64")
            u_dy = tf.cast(tf.signal.ifft2d(waves_x * 1j * omega_y), "float64")
            v_dx = tf.cast(tf.signal.ifft2d(waves_y * 1j * omega_x), "float64")
            v_dy = tf.cast(tf.signal.ifft2d(waves_y * 1j * omega_y), "float64")

            u_new = u - (u*u_dx + v*u_dy) * 0.1
            v_new = v - (u*v_dx + v*v_dy) * 0.1
            return tf.expand_dims(tf.stack([u_new, v_new], axis=-1), 0)

        if advection_method == "warp":
            self.self_advection = tf.function(self_advection_warp)
        elif advection_method == "gradient":
            self.self_advection = tf.function(self_advection_gradient)
        else:
            raise ValueError("Valid advection methods are 'warp' and 'gradient'")
        self.viscosity_and_mass_conservation = tf.function(viscosity_and_mass_conservation)

    def step(self):
        if self.force_function is not None:
            # Acceleration is proportional to force.
            # Acceleration is the same as change in velocity (flow).
            self.flow += force_function(self.t)
        self.flow = self.self_advection(self.flow)
        self.flow = self.viscosity_and_mass_conservation(self.flow)
        self.t += 1

    def numpy(self):
        return tf.squeeze(self.flow).numpy()


if __name__ == '__main__':
    from pylab import *
    from matplotlib.animation import FuncAnimation
    x = np.linspace(-4, 4, 81)[:-1]
    x, y = np.meshgrid(x, x)

    flow = randn(x.shape[0], x.shape[1], 2) * 0.5

    def source_function(t):
        x0 = 3*sin(t*0.03) + cos(t*0.0123)*0.1
        y0 = 3*cos(t*0.0242)
        direction = t*0.01
        return x0, y0, direction

    def force_function(t):
        x0, y0, direction = source_function(t)
        force = exp(-(x-x0)**2-(y-y0)**2) * 0.05
        force = [cos(direction)*force, sin(direction)*force]
        return np.swapaxes(force, 0, 2)

    solver = FluidSolver(flow, force_function=force_function, advection_method="gradient")
    solver.step()  # Get rid of flow divergence

    flow = solver.numpy()
    x0, y0, direction = source_function(solver.t)
    plots = plot([x0], [y0], "o")
    plots.extend(plot([x0+cos(direction)*0.1], [y0+sin(direction)*0.1], "."))
    plots.append(quiver(x[::2,::2], y[::2,::2], flow[::2,::2,0], flow[::2,::2,1]))

    def update(frame):
        solver.step()
        flow = solver.numpy()
        x0, y0, direction = source_function(solver.t)
        plots[0].set_xdata([x0])
        plots[0].set_ydata([y0])
        plots[1].set_xdata([x0+cos(direction)*0.1])
        plots[1].set_ydata([y0+sin(direction)*0.1])
        plots[2].set_UVC(flow[::2,::2,0], flow[::2,::2,1])
        return plots

    FuncAnimation(gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    show()
