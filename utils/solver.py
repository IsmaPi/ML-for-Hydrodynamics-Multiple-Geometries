"""Shared libMobility solver creation and callable wrapper.

Single source of truth for solver initialization across the project.
"""

import numpy as np


def create_solver(geometry: str, viscosity: float, a: float, box_size: float = 32.0):
    """Create and initialize a libMobility solver.

    Args:
        geometry: "nbody_open" or "pse_periodic"
        viscosity: fluid viscosity
        a: hydrodynamic radius
        box_size: box size for periodic geometry

    Returns:
        Initialized libMobility solver.
    """
    import libMobility as lm

    if geometry == "nbody_open":
        solver = lm.NBody(periodicityX="open", periodicityY="open", periodicityZ="open")
        solver.initialize(viscosity=viscosity, hydrodynamicRadius=a)
    elif geometry == "pse_periodic":
        solver = lm.PSE(periodicityX="periodic", periodicityY="periodic", periodicityZ="periodic")
        solver.setParameters(Lx=box_size, Ly=box_size, Lz=box_size, psi=1.0)
        solver.initialize(viscosity=viscosity, hydrodynamicRadius=a)
    else:
        raise ValueError(f"Unknown geometry: {geometry}. Use 'nbody_open' or 'pse_periodic'.")

    return solver


class SolverCallable:
    """Wraps a libMobility solver as a callable: (positions, forces) -> displacement.

    Args:
        geometry: "nbody_open" or "pse_periodic"
        viscosity: fluid viscosity
        a: hydrodynamic radius
        box_size: box size for periodic geometry
    """

    def __init__(self, geometry: str, viscosity: float, a: float, box_size: float = 32.0):
        self.solver = create_solver(geometry, viscosity, a, box_size)

    def __call__(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Compute displacement = M(X) . F.

        Args:
            positions: (N, 3) particle positions
            forces: (N, 3) per-particle forces

        Returns:
            (N, 3) displacement array
        """
        N = positions.shape[0]
        self.solver.setPositions(positions)
        disp, _ = self.solver.Mdot(forces=forces)
        return np.array(disp, dtype=np.float32).reshape(N, 3)

    def clean(self):
        """Release solver resources."""
        self.solver.clean()
