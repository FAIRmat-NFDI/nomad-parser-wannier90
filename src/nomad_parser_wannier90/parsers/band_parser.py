#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from typing import Optional, List
from structlog.stdlib import BoundLogger

from nomad.parsing.file_parser import DataTextParser

from nomad_simulations.schema.model_method import Wannier
from nomad_simulations.schema.model_system import ModelSystem
from nomad_simulations.schema.numerical_settings import (
    KSpace,
    KLinePath as KLinePathSettings,
)
from nomad_simulations.schema.properties import ElectronicBandStructure
from nomad_simulations.schema.variables import KLinePath


class Wannier90BandParser:
    def __init__(self, band_file: str = ''):
        if not band_file:
            raise ValueError('Band structure `*band.dat` file not found.')
        self.band_parser = DataTextParser(mainfile=band_file)

    def parse_k_line_path_settings(
        self,
        reciprocal_lattice_vectors: Optional[np.ndarray],
        k_line_path: KLinePathSettings,
        logger: BoundLogger,
    ) -> None:
        """
        Parse the `KLinePath` settings from the `*band.dat` file using the `KLinePath.resolve_points` method. The
        information is then stored under `KLinePath.points`, and will be used to extract the `ElectronicBandStructure` variables
        points.

        Args:
            reciprocal_lattice_vectors (Optional[np.ndarray]): The reciprocal lattice vectors.
            k_line_path (KLinePathSettings): The `KLinePath` settings section
            logger (BoundLogger): The logger to log messages.
        """
        try:
            kpath_norms = self.band_parser.data.transpose()[0]
            k_line_path.resolve_points(
                points_norm=kpath_norms,
                reciprocal_lattice_vectors=reciprocal_lattice_vectors,
                logger=logger,
            )
        except Exception:
            logger.info('Error parsing `KLinePath` settings.')

    def parse_band_structure(
        self,
        wannier_method: Optional[Wannier],
        k_space: Optional[KSpace],
        model_systems: List[ModelSystem],
        logger: BoundLogger,
    ) -> Optional[ElectronicBandStructure]:
        if wannier_method is None:
            logger.warning('Could not parse the `Wannier` method.')
            return None, None
        n_orbitals = wannier_method.n_orbitals
        if k_space is None:
            logger.info('`KSpace` settings not found.')
            return None

        # Resolving `reciprocal_lattice_vectors` from `KSpace` method
        rlv = k_space.resolve_reciprocal_lattice_vectors(model_systems, logger)
        # And parsing the points from the `*band.dat` file
        k_line_path = k_space.k_line_path
        self.parse_k_line_path_settings(
            reciprocal_lattice_vectors=rlv,
            k_line_path=k_line_path,
            logger=logger,
        )
        if k_line_path.points is None:
            logger.warning('Could not resolve the `KLinePath` points.')
            return None

        # Parse the band structure data
        band_structure = ElectronicBandStructure(n_bands=n_orbitals)
        if self.band_parser.data is None:
            logger.warning('Could not parse the band structure data.')
            return None
        k_line_path_variables = KLinePath()
        k_line_path_variables.points = k_line_path  # ! Wait for @amirgolpv to check
        band_structure.variables = [k_line_path_variables]
        data = self.band_parser.data.transpose()[1:].transpose()
        band_structure.value = data
        return band_structure
