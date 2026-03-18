from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from app.models.svi.common import MarketSlice, CalibrationResult
from app.services.svi.raw_svi import RawSVIPlottingService


class RawSVIResultStore:
    def __init__(self, base_output_dir: str = "outputs/svi/raw"):
        self.base_output_dir = Path(base_output_dir)

    def ensure_output_dir(self) -> Path:
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        return self.base_output_dir

    def save_result_json(
        self,
        calibration_result: CalibrationResult,
        filename: str = "calibration_result.json",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(calibration_result.to_dict(), f, indent=4)

        return file_path

    def save_comparison_csv(
        self,
        calibration_result: CalibrationResult,
        filename: str = "market_vs_fitted.csv",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        df = pd.DataFrame({
            "strike": calibration_result.strikes,
            "log_moneyness": calibration_result.log_moneyness,
            "market_iv": calibration_result.market_iv,
            "fitted_iv": calibration_result.fitted_iv,
            "market_total_variance": calibration_result.market_total_variance,
            "fitted_total_variance": calibration_result.fitted_total_variance,
        })

        df.to_csv(file_path, index=False)
        return file_path

    def save_total_variance_plot(
        self,
        market_slice: MarketSlice,
        calibration_result: CalibrationResult,
        filename: str = "total_variance_fit.png",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        RawSVIPlottingService.plot_total_variance(
            market_slice=market_slice,
            calibration_result=calibration_result,
        )
        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_implied_vol_plot(
        self,
        market_slice: MarketSlice,
        calibration_result: CalibrationResult,
        filename: str = "implied_vol_fit.png",
    ) -> Path:
        output_dir = self.ensure_output_dir()
        file_path = output_dir / filename

        RawSVIPlottingService.plot_implied_vol(
            market_slice=market_slice,
            calibration_result=calibration_result,
        )
        plt.savefig(file_path, bbox_inches="tight", dpi=150)
        plt.close()

        return file_path

    def save_all(
        self,
        market_slice: MarketSlice,
        calibration_result: CalibrationResult,
        prefix: Optional[str] = None,
    ) -> dict:
        prefix_str = f"{prefix}_" if prefix else ""

        json_path = self.save_result_json(
            calibration_result,
            filename=f"{prefix_str}calibration_result.json",
        )
        csv_path = self.save_comparison_csv(
            calibration_result,
            filename=f"{prefix_str}market_vs_fitted.csv",
        )
        total_variance_plot_path = self.save_total_variance_plot(
            market_slice,
            calibration_result,
            filename=f"{prefix_str}total_variance_fit.png",
        )
        implied_vol_plot_path = self.save_implied_vol_plot(
            market_slice,
            calibration_result,
            filename=f"{prefix_str}implied_vol_fit.png",
        )

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "total_variance_plot": str(total_variance_plot_path),
            "implied_vol_plot": str(implied_vol_plot_path),
        }