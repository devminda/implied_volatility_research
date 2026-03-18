import numpy as np

from app.models.svi.surface_svi.phi.heston_like import HestonLikePhi
from app.models.svi.surface_svi.model import SSVIFormula
from app.services.svi.surface_svi.results_store import SSVIResultStore


def main():
    rho = -0.7
    phi = HestonLikePhi(lam=2.0)

    model = SSVIFormula(rho=rho, phi=phi)

    k = np.linspace(-1.0, 1.0, 100)
    thetas = np.linspace(0.01, 0.20, 80)

    maturity = 0.5

    store = SSVIResultStore()

    outputs = store.save_all(
        k_grid=k,
        thetas=thetas,
        ssvi_model=model,
        maturity=maturity,
        prefix="heston_like",
    )

    print(outputs)


if __name__ == "__main__":
    main()