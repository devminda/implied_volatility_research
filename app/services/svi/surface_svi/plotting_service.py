from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class SSVIPlottingService:

    @staticmethod
    def plot_smiles(
        k_grid,
        thetas,
        ssvi_model,
        maturity: float,
        show: bool = True,
    ):
        """
        Plot implied volatility smiles for multiple theta values.
        """
        for theta in thetas:
            w = ssvi_model.evaluate(k_grid, theta)
            iv = np.sqrt(np.maximum(w, 0.0) / maturity)

            plt.plot(k_grid, iv, label=f"theta={theta:.3f}")

        plt.xlabel("log-moneyness k = ln(K/F)")
        plt.ylabel("Implied Volatility")
        plt.title("SSVI Implied Volatility Smiles")
        plt.legend()
        plt.grid(True)

        if show:
            plt.show()

    @staticmethod
    def plot_total_variance(
        k_grid,
        thetas,
        ssvi_model,
        show: bool = True,
    ):
        """
        Plot total variance curves.
        """
        for theta in thetas:
            w = ssvi_model.evaluate(k_grid, theta)
            plt.plot(k_grid, w, label=f"theta={theta:.3f}")

        plt.xlabel("log-moneyness k = ln(K/F)")
        plt.ylabel("Total Variance")
        plt.title("SSVI Total Variance Curves")
        plt.legend()
        plt.grid(True)

        if show:
            plt.show()
    
    @staticmethod
    def plot_total_variance_surface(
        k_grid,
        theta_grid,
        ssvi_model,
        show: bool = True,
    ) -> None:
        """
        Plot the SSVI total variance surface w(k, theta).

        Parameters
        ----------
        k_grid : array-like
            Log-moneyness grid.
        theta_grid : array-like
            ATM total variance grid.
        ssvi_model : object
            Model with evaluate(k, theta) method.
        show : bool
            Whether to display the plot immediately.
        """
        k_grid = np.asarray(k_grid, dtype=float)
        theta_grid = np.asarray(theta_grid, dtype=float)

        kk, tt = np.meshgrid(k_grid, theta_grid)

        w_surface = np.zeros_like(kk, dtype=float)

        for i in range(tt.shape[0]):
            for j in range(tt.shape[1]):
                w_surface[i, j] = ssvi_model.evaluate(kk[i, j], tt[i, j])

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(kk, tt, w_surface, cmap="viridis")

        ax.set_xlabel("Log-moneyness k = ln(K/F)")
        ax.set_ylabel("ATM total variance theta")
        ax.set_zlabel("Total variance w(k, theta)")
        ax.set_title("SSVI Total Variance Surface")

        if show:
            plt.show()