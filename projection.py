"""Projection
=============

The projection information, pϑ(ξ) is, at its core, a function ℝ² → ℝ.  Within
the computer, this will be stored as ℕ² → ℝ, where ℝ is approximated by a
float.

Instead of defining a whole lot of different functions to handle the 2D array
(and then needing to remember the number of steps in theta, or the range of xi
values), we use a struct to store everything together.

"""
import math as m
import numpy as np
import scipy as sc


class Projection():
    """The class to store all the information pertaining to a particular
    projection.

    Internally, the data is stored in a 2D array, with the size of the array
    determining the number of steps in theta and xi that are used.  The range
    in theta and xi is also stored within the struct and all functions that
    work on projections can be implemented as methods.

    """

    def __init__(self, theta_range, theta_steps, xi_range, xi_steps):
        """Create a new empty projection object.

        The range of theta values and corresponding number of subdivisions are
        specified in `(theta_min, theta_max)` and `theta_steps`.  Similarly,
        the range of xi values and corresponding number of subdivisions are
        specified in `(xi_min, xi_max)` and `xi_steps`.

        """
        # Make sure that theta_min and theta_max are correctly ordered, and
        # calculate the step size.  Then repeat with xi.
        self.theta_min = min(theta_range)
        self.theta_max = max(theta_range)
        self.theta_steps = theta_steps
        self.theta_step_size = (self.theta_max - self.theta_min) / theta_steps

        self.xi_min = min(xi_range)
        self.xi_max = max(xi_range)
        self.xi_steps = xi_steps
        self.xi_step_size = (self.xi_max - self.xi_min) / xi_steps

        self.data = np.zeros((theta_steps, xi_steps)) #Return a new array of (theta_steps, xi_steps), filled with zeros.

    def theta(self, idx):
        """Return the value of theta corresponding to a particular index."""
        return self.theta_min + idx * self.theta_step_size

    def xi_func(self, idx):
        """Return xi at a particular step index."""
        return self.xi_min + idx * self.xi_step_size

    def get_continuous(self, theta_idx, xi):
        """Return the value of the projection at any arbitrary point using
        interpolation (for a given value of theta).

        This allows for the projection to be defined for all values in the
        range of xi values used in the projection. For values of xi which are
        outside of the range in the projection, the edge point is returned:

        ```text
            |             XXX           |
            |            X   X          |
     XXXXXXXX           X     X         |
            |X XX       X     X         |
            | X  X     X       X        |
            |     X   X         X       |
            |      XXX           X      XXXXXXXX
            |                    X    XX|
            |                     X  X  |
            |                      XX   |
        ```

        """
        return self.get_continuous_custom(theta_idx, xi, self.data)

    def get_continuous_custom(self, theta_idx, xi, data):
        """This functions behaves exactly like `get_continuous`, but instead of
        using the data array contained within the projection, an other array can
        be specified (which must be of the same shape).

        This is intended to be used internally when manipulating data that raise NotImplementedError()s
        and has been filtered and Fourier transformed.

        """
        xi_idx = (xi - self.xi_min) / self.xi_step_size
        if xi_idx < 0:
            return data[theta_idx, 0]
        elif xi_idx >= self.xi_steps:
            return data[theta_idx, -1]
        else:
            return data[theta_idx, int(xi_idx)]

    def add_object(self, obj, eta_range):
        """Add an object to the projection.
        """

        self.objects = obj
        self.eta_range = eta_range
        self.pa_to_array(obj)



    def pa_to_array(self, obj):
        for idx in np.ndindex(self.data.shape):
            self.data[idx] += obj.project_attenuation(self.theta(idx[0]),self.xi_func(idx[1]),self.eta_range)




    def back_project(self):
        """Reconstruct the original image by back-projection.

        The resulting array size is square and has the same width as the number
        of xi steps in the original projection.

        """
        self.original_image = sc.integrate.quad(lambda x: special.self.data, 0, m.pi)

        self.add_object(original_image, self.eta_range)



        raise NotImplementedError()

    def filtered_back_project(self, f):
        """Reconstruct the original image by filtered back-projection.

        The resulting array size is square and has the same width as the number
        of xi steps in the original projection.

        The filter is an arbitrary function on floats and acts on the
        wavenumber.

        """
        raise NotImplementedError()
