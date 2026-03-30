from __future__ import annotations

from app.models.svi.surface_svi.model import SSVIFormula
import numpy as np
import matplotlib.pyplot as plt


class SSVIPlottingService:

    @staticmethod
    def plot_smiles_by_maturity(
        k_grid,
        maturities,
        theta_of_t,
        ssvi_model,
        show: bool = True,
    ):
        k_grid    = np.asarray(k_grid, dtype=float)
        maturities = np.asarray(maturities, dtype=float)

        # Use build_iv_surface for consistency — avoids duplicating the
        # theta->w->iv logic that already lives in the model
        kk, TT, w, iv = SSVIFormula.build_iv_surface(
            k_grid=k_grid,
            T_grid=maturities,
            model=ssvi_model,
            theta_func=theta_of_t,
        )

        plt.figure(figsize=(10, 6))

        for i, T in enumerate(maturities):
            theta = float(theta_of_t(T))
            plt.plot(k_grid, iv[i, :], label=f"T={T:.2f}, θ={theta:.3f}")

        plt.xlabel("Log-moneyness k = ln(K/F)")
        plt.ylabel("Implied Volatility")
        plt.title("SSVI Implied Volatility Smiles")
        plt.legend(fontsize=7)
        plt.grid(True)
        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def plot_total_variance(
        k_grid,
        thetas,
        ssvi_model,
        show: bool = True,
    ):
        k_grid = np.asarray(k_grid, dtype=float)
        thetas = np.asarray(thetas, dtype=float)

        plt.figure(figsize=(10, 6))

        for theta in thetas:
            w = ssvi_model.evaluate(k_grid, theta)
            plt.plot(k_grid, w, label=f"θ={theta:.3f}")

        plt.xlabel("Log-moneyness k = ln(K/F)")
        plt.ylabel("Total Variance")
        plt.title("SSVI Total Variance Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def plot_implied_vol_surface(
        k_grid,
        maturity_grid,
        theta_of_t,
        ssvi_model,
        show: bool = True,
        cmap: str = "turbo",
    ) -> None:
        """
        Plot implied volatility surface sigma_BS(k, T).
        """
        k_grid = np.asarray(k_grid, dtype=float)
        maturity_grid = np.asarray(maturity_grid, dtype=float)

        kk, TT, w, iv_surface = SSVIFormula.build_iv_surface(
            k_grid=k_grid,
            T_grid=maturity_grid,
            model=ssvi_model,
            theta_func=theta_of_t,
        )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            kk,
            TT,
            iv_surface,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
        )

        ax.set_xlabel("Log-moneyness k = ln(K/F)", labelpad=12)
        ax.set_ylabel("Maturity T", labelpad=12)
        ax.set_zlabel("Implied Volatility σ_BS(k, T)", labelpad=14)
        ax.set_title("SSVI Implied Volatility Surface", pad=18)

        ax.view_init(elev=25, azim=-60)

        fig.colorbar(surf, ax=ax, shrink=0.7, aspect=18, pad=0.08)

        plt.tight_layout()

        if show:
            plt.show()

    @staticmethod
    def plot_total_variance_surface_in_theta_space(
        k_grid,
        theta_grid,
        ssvi_model,
        show: bool = True,
        cmap: str = "turbo",
    ) -> None:
        """
        Plot the SSVI total variance surface w(k, theta).
        This is the mathematical SSVI surface in theta-space.
        """
        k_grid = np.asarray(k_grid, dtype=float)
        theta_grid = np.asarray(theta_grid, dtype=float)

        kk, theta_mesh = np.meshgrid(k_grid, theta_grid)
        w_surface = np.zeros_like(kk, dtype=float)

        for i in range(theta_mesh.shape[0]):
            w_surface[i, :] = ssvi_model.evaluate(k_grid, theta_mesh[i, 0])

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            kk,
            theta_mesh,
            w_surface,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
        )

        ax.set_xlabel("Log-moneyness k = ln(K/F)", labelpad=12)
        ax.set_ylabel("ATM Total Variance θ", labelpad=12)
        ax.set_zlabel("Total Variance w(k, θ)", labelpad=14)
        ax.set_title("SSVI Total Variance Surface in θ-space", pad=18)

        ax.view_init(elev=25, azim=-60)

        fig.colorbar(surf, ax=ax, shrink=0.7, aspect=18, pad=0.08)

        plt.tight_layout()

        if show:
            plt.show()