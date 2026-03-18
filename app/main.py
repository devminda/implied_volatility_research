from app.models.svi.common import MarketSlice
from app.services.svi.raw_svi import RawSVICalibrationService, RawSVIResultStore


def main() -> None:
    market_slice = MarketSlice(
    spot=100.0,
    rate=0.03,
    dividend_yield=0.01,
    maturity=0.5,
    strikes=[
        70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90, 92.5,
        95, 97.5, 100, 102.5, 105, 107.5, 110, 112.5, 115,
        117.5, 120, 122.5, 125, 127.5, 130
    ],
    implied_vols=[
        0.340, 0.330, 0.320, 0.310, 0.300, 0.290, 0.280, 0.270, 0.260, 0.250,
        0.240, 0.228, 0.220, 0.221, 0.225, 0.229, 0.235, 0.240, 0.245,
        0.250, 0.255, 0.261, 0.268, 0.274, 0.280
    ],
)

    calibration_service = RawSVICalibrationService()
    result = calibration_service.calibrate(market_slice)

    print("Calibration finished.")
    print(result.to_dict())

    result_store = RawSVIResultStore(base_output_dir="outputs/svi/raw")
    saved_files = result_store.save_all(
        market_slice=market_slice,
        calibration_result=result,
        prefix="example_run",
    )

    print("\nSaved files:")
    for name, path in saved_files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()