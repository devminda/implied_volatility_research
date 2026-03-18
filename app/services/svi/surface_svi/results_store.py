from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.services.svi.surface_svi.plotting_service import SSVIPlottingService


class SSVIResultStore:
    def __init__(self, base_output_dir: str = "outputs/svi/ssvi"):
        self.base_output_dir = Path(base_output_dir)

    def ensure_output_dir(self) -> Path:
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        return self.base_output_dir

    
    def save_smile_csv(
        self,
        k_grid,
        thetas,
        ssvi_model,
        maturity: float,
        filename: str = "ssvi_smiles.csv",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        data = []

        for theta in thetas:
            w = ssvi_model.evaluate(k_grid, theta)
            iv = np.sqrt(np.maximum(w, 0.0) / maturity)

            for k, w_val, iv_val in zip(k_grid, w, iv):
                data.append({
                    "theta": theta,
                    "log_moneyness": k,
                    "total_variance": w_val,
                    "implied_vol": iv_val,
                })

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        return file_path

    
    def save_total_variance_plot(
        self,
        k_grid,
        thetas,
        ssvi_model,
        filename: str = "total_variance.png",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_total_variance(
            k_grid=k_grid,
            thetas=thetas,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    
    def save_implied_vol_plot(
        self,
        k_grid,
        thetas,
        ssvi_model,
        maturity: float,
        filename: str = "implied_vol.png",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_smiles(
            k_grid=k_grid,
            thetas=thetas,
            ssvi_model=ssvi_model,
            maturity=maturity,
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
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        SSVIPlottingService.plot_total_variance_surface(
            k_grid=k_grid,
            theta_grid=theta_grid,
            ssvi_model=ssvi_model,
            show=False,
        )

        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_all(
        self,
        k_grid,
        thetas,
        ssvi_model,
        maturity: float,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        prefix_str = f"{prefix}_" if prefix else ""

        csv_path = self.save_smile_csv(
            k_grid,
            thetas,
            ssvi_model,
            maturity,
            filename=f"{prefix_str}smiles.csv",
        )

        total_variance_plot = self.save_total_variance_plot(
            k_grid,
            thetas,
            ssvi_model,
            filename=f"{prefix_str}total_variance.png",
        )

        implied_vol_plot = self.save_implied_vol_plot(
            k_grid,
            thetas,
            ssvi_model,
            maturity,
            filename=f"{prefix_str}implied_vol.png",
        )

        total_variance_surface = self.save_total_variance_surface(
            k_grid,
            thetas,
            ssvi_model,
            filename=f"{prefix_str}total_variance_surface.png",
        )

        return {
            "csv": str(csv_path),
            "total_variance_plot": str(total_variance_plot),
            "implied_vol_plot": str(implied_vol_plot),
            "total_variance_surface": str(total_variance_surface),
        }