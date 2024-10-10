from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

import numpy as np
from nomad.parsing.file_parser import Quantity, TextParser
from nomad.units import ureg
from nomad_simulations.schema_packages.model_method import Wannier
from nomad_simulations.schema_packages.properties import (
    CrystalFieldSplitting,
    HoppingMatrix,
)
from nomad_simulations.schema_packages.variables import WignerSeitz


class HrParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity('degeneracy_factors', r'\s*written on[\s\w]*:\d*:\d*\s*([\d\s]+)'),
            Quantity('hoppings', r'\s*([-\d\s.]+)', repeats=False),
        ]


class Wannier90HrParser:
    def __init__(self, hr_file: str = ''):
        if not hr_file:
            raise ValueError('Hopping `*hr.dat` file not found.')
        self.hr_parser = HrParser(mainfile=hr_file)

    def parse_hoppings(
        self, wannier_method: Optional[Wannier], logger: 'BoundLogger'
    ) -> tuple[Optional[HoppingMatrix], Optional[CrystalFieldSplitting]]:
        """
        Parse the `HoppingMatrix` and `CrystalFieldSplitting` sections from the `*hr.dat` file.

        Args:
            wannier_method (Wannier): The `Wannier` method section which contains the number of orbitals information.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (tuple[Optional[HoppingMatrix], Optional[CrystalFieldSplitting]]): The parsed `HoppingMatrix` and
            `CrystalFieldSplitting` properties].
        """
        if wannier_method is None:
            logger.warning('Could not parse the `Wannier` method.')
            return None, None
        n_orbitals = wannier_method.n_orbitals

        # Parsing the `HoppingMatrix` and `CrystalFieldSplitting` sections
        crystal_field_splitting = CrystalFieldSplitting(
            n_orbitals=n_orbitals, variables=[]
        )
        hopping_matrix = HoppingMatrix(n_orbitals=n_orbitals)

        # Parse the `degeneracy_factors` and `value` of the `HoppingMatrix`
        deg_factors = self.hr_parser.get('degeneracy_factors', [])
        if deg_factors is None or len(deg_factors) == 0:
            logger.warning('Could not parse the degeneracy factors.')
            return None, None

        # Define the variables `WignerSeitz`
        n_wigner_seitz_points = deg_factors[1]
        wigner_seitz = WignerSeitz(
            n_points=n_wigner_seitz_points
        )  # delete crystal field Wigner-Seitz point from `HoppingMatrix` variable
        hopping_matrix.variables.append(wigner_seitz)

        # deg_factors and full_hoppings contain both the `CrystalFieldSplitting` and `HoppingMatrix` values
        degeneracy_factors = deg_factors[2:]
        full_hoppings = self.hr_parser.get('hoppings', [])
        try:
            hops = np.reshape(
                full_hoppings,
                (n_wigner_seitz_points, n_orbitals, n_orbitals, 7),
            )

            # storing the crystal field splitting values
            ws0 = int((n_wigner_seitz_points - 1) / 2)
            crystal_fields = [
                hops[ws0, i, i, 5] for i in range(n_orbitals)
            ]  # only real elements
            crystal_field_splitting.value = crystal_fields * ureg('eV')

            # delete repeated points for different orbitals
            ws_points = hops[:, :, :, :3]
            ws_points = np.unique(ws_points.reshape(-1, 3), axis=0)
            wigner_seitz.points = ws_points

            # passing hoppings
            hoppings = hops[:, :, :, -2] + 1j * hops[:, :, :, -1]
            hopping_matrix.value = hoppings * ureg('eV')
            hopping_matrix.degeneracy_factors = degeneracy_factors
        except Exception:
            logger.warning('Could not parse the hopping matrix values.')
            return None, None
        return hopping_matrix, crystal_field_splitting
