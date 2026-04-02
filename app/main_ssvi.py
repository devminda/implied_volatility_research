import numpy as np

from app.models.svi.surface_svi.phi.heston_like import HestonLikePhi, PowerLawPhi, StabilizedPowerLawPhi
from app.models.svi.surface_svi.model import SSVIFormula
from app.services.svi.surface_svi.results_store import SSVIResultStore


def run_example(prefix: str, phi, rho: float = -0.7, n_k: int = 100):
    model = SSVIFormula(rho=rho, phi=phi)

    k_grid     = np.linspace(-1.0, 1.0, 20)
    T_grid     = np.linspace(0.1, 2.0, 20)
    theta_grid = np.linspace(0.01, 0.20, 20)

    theta_func = lambda T: SSVIFormula.theta_of_T(T, sigma_atm=0.20) 

    store = SSVIResultStore()
    outputs = store.save_all(
        k_grid=k_grid,
        theta_grid=theta_grid,
        maturity_grid=T_grid,
        theta_of_t=theta_func,
        ssvi_model=model,
        prefix=prefix,
    )
    print(outputs)


def main():
    run_example("heston_like",          HestonLikePhi(lam=2.0))
    run_example("power_law",            PowerLawPhi(eta=1.0, gamma=0.8),            n_k=150)
    run_example("stabilized_power_law", StabilizedPowerLawPhi(eta=0.3, gamma=0.8),  n_k=150)
    eta = 0.3
    rho = -0.4
    print(eta * (1 + abs(rho)))

if __name__ == "__main__":
    main()