from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.services.svi.surface_svi.plotting_service import SSVIPlottingService
from app.models.svi.surface_svi.model import SSVIFormula


class SSVIResultStore:
    def __init__(self, base_output_dir: str = "outputs/svi/ssvi"):
        self.base_output_dir = Path(base_output_dir)

    def ensure_output_dir(self) -> Path:
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        return self.base_output_dir

    def save_surface_csv(
        self,
        k_grid,
        maturity_grid,
        theta_of_t: Callable[[float], float],
        ssvi_model,
        filename: str = "ssvi_surface.csv",
    ) -> Path:
        """
        Single unified CSV with all columns:
            maturity, theta, log_moneyness, total_variance, implied_vol
        """
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        k_grid       = np.asarray(k_grid, dtype=float)
        maturity_grid = np.asarray(maturity_grid, dtype=float)

        kk, TT, w, iv = SSVIFormula.build_iv_surface(
            k_grid=k_grid,
            T_grid=maturity_grid,
            model=ssvi_model,
            theta_func=theta_of_t,
        )

        theta_mesh = theta_of_t(TT)  # same shape as TT

        rows = {
            "maturity":       TT.ravel(),
            "theta":          theta_mesh.ravel(),
            "log_moneyness":  kk.ravel(),
            "total_variance": w.ravel(),
            "implied_vol":    iv.ravel(),
        }

        pd.DataFrame(rows).to_csv(file_path, index=False)
        return file_path



    def save_implied_vol_plot(
        self,
        k_grid,
        maturities,
        theta_of_t: Callable[[float], float],
        ssvi_model,
        filename: str = "implied_vol.png",
    ) -> Path:
        """
        Save implied volatility smiles by maturity.
        """
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_smiles_by_maturity(
            k_grid=k_grid,
            maturities=maturities,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_total_variance_surface(
        self,
        k_grid,
        theta_grid,
        ssvi_model,
        filename: str = "total_variance_surface.png",
    ) -> Path:
        """
        Save total variance surface in theta-space:
            w(k, theta)
        """
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_total_variance_surface_in_theta_space(
            k_grid=k_grid,
            theta_grid=theta_grid,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_implied_vol_surface(
        self,
        k_grid,
        maturity_grid,
        theta_of_t: Callable[[float], float],
        ssvi_model,
        filename: str = "implied_vol_surface.png",
    ) -> Path:
        """
        Save implied volatility surface in maturity-space:
            sigma_BS(k, T)
        """
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_implied_vol_surface(
            k_grid=k_grid,
            maturity_grid=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_total_variance_plot(
        self,
        k_grid,
        theta_grid,
        ssvi_model,
        filename: str = "total_variance.png",
    ) -> Path:
        """
        Save total variance curves in theta-space.
        """
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_total_variance(
            k_grid=k_grid,
            thetas=theta_grid,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_local_vol_surface(
        self,
        k_grid,
        maturity_grid,
        theta_of_t,
        ssvi_model,
        filename: str = "local_vol_surface.png",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path  = output_dir / filename

        SSVIPlottingService.plot_local_vol_surface(
            k_grid=k_grid,
            maturity_grid=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path
    
    def save_all(
        self,
        k_grid,
        theta_grid,
        maturity_grid,
        theta_of_t: Callable[[float], float],
        ssvi_model,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        prefix_str = f"{prefix}_" if prefix else ""

        surface_csv = self.save_surface_csv(
            k_grid=k_grid,
            maturity_grid=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}surface.csv",
        )

        total_variance_plot = self.save_total_variance_plot(
            k_grid=k_grid,
            theta_grid=theta_grid,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}total_variance.png",
        )

        implied_vol_plot = self.save_implied_vol_plot(
            k_grid=k_grid,
            maturities=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}implied_vol.png",
        )

        total_variance_surface = self.save_total_variance_surface(
            k_grid=k_grid,
            theta_grid=theta_grid,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}total_variance_surface.png",
        )

        implied_vol_surface = self.save_implied_vol_surface(
            k_grid=k_grid,
            maturity_grid=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}implied_vol_surface.png",
        )

        local_vol_surface = self.save_local_vol_surface(
            k_grid=k_grid,
            maturity_grid=maturity_grid,
            theta_of_t=theta_of_t,
            ssvi_model=ssvi_model,
            filename=f"{prefix_str}local_vol_surface.png",
        )

        return {
            "surface_csv":            str(surface_csv),
            "total_variance_plot":    str(total_variance_plot),
            "implied_vol_plot":       str(implied_vol_plot),
            "total_variance_surface": str(total_variance_surface),
            "implied_vol_surface":    str(implied_vol_surface),
            "local_vol_surface":      str(local_vol_surface),
        }