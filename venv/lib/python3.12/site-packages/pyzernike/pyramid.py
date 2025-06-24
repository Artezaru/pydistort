import numpy
import numbers
from .zernike_polynomial import zernike_polynomial, radial_polynomial
import matplotlib.pyplot as plt
from matplotlib import cm

def pyramid(N: int = 5, radial: bool = False, rho_derivative: int = 0, theta_derivative: int = 0, close: bool = False):
        """
        Plots Zernike polynomials up to a given order N in a pyramid layout.

        Parameters
        ----------
        N : int, optional
            The maximum order of Zernike polynomials to plot. The default is 5.
        radial : bool, optional
            If True, plot the radial Zernike polynomials. The default is False
        rho_derivative : int, optional
            The order of the derivative with respect to rho. The default is 0.
        theta_derivative : int, optional
            The order of the derivative with respect to theta. The default is 0.
        close : bool, optional
            If True, close all open figures before plotting. The default is False.

        Raises
        ------
        ValueError
            If N is not a non-negative integer.
        """
        if not isinstance(N, numbers.Integral) or N < 0:
            raise ValueError("N must be a non-negative integer.")

        # Close all open figures if requested
        if close:
            plt.close('all')

        # Create a new figure
        fig = plt.figure(figsize=(14, 10))
        fs = 10
        fs1 = 8

        x =  numpy.linspace(-1.2, 1.2, 400) 
        y =  numpy.linspace(-1.2, 1.2, 400)
        X, Y =  numpy.meshgrid(x, y)
        rho =  numpy.sqrt(X**2 + Y**2)
        theta =  numpy.arctan2(Y, X)

        span = 0.03  # Reduced span to minimize empty space
        leftoff = 0.02
        nradial = N

        # Create a colormap normalization for consistent color scaling
        norm = cm.colors.Normalize(vmin=-1, vmax=1)
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.seismic)

        while nradial >= 0:
            nk = (nradial + 1) * (nradial + 2) // 2
            nrows = nradial + 1
            ncols = 2 * nradial + 1
            height1 = (1 - (nrows + 1) * span) / nrows
            width1 = (1 - (ncols + 1) * span) / ncols
            min1 = min(width1, height1)
            if min1 > 0:
                height1 = min1
                width1 = min1
                width_span = (1 - min1 * ncols) / (ncols + 1)
                height_span = (1 - min1 * nrows) / (nrows + 1)
                break
            else:
                nradial -= 1

        for n in range(nradial + 1):
            m_values =  numpy.arange(-n, n + 1, 2).astype(int)
            left = (1 - len(m_values) * width1 - (len(m_values) - 1) * width_span) / 2
            bott = (1 - nrows * height1 - (nrows - 1) * height_span) / 2
            bt = bott + (nrows - n - 1) * (height1 + height_span)

            for idx, m in enumerate(m_values):
                lf = left + idx * (width1 + width_span) + leftoff
                ax = fig.add_axes([lf, bt, width1, height1])

                m= int(m)
                n = int(n)

                if radial:
                    Z = radial_polynomial(rho, n, m, rho_derivative=rho_derivative, default= numpy.nan)
                else:
                    Z = zernike_polynomial(rho, theta, n, m, rho_derivative=rho_derivative, theta_derivative=theta_derivative, default= numpy.nan)

                im = ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap=cm.seismic, norm=norm)
                ax.axis('off')
                ax.set_title(f"n={n}, m={m}", fontsize=fs1)

        # Adding a colorbar to the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plt.colorbar(mappable, cax=cbar_ax, orientation='vertical')

        plt.text(0.5, 1.05, '$n$', transform=fig.transFigure, ha='center', fontsize=fs)
        plt.text(-0.05, 0.5, '$m$', transform=fig.transFigure, va='center', rotation='vertical', fontsize=fs)
        plt.show()