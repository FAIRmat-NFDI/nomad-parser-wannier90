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

from typing import Optional

import numpy as np
from nomad.parsing.file_parser import DataTextParser
from nomad.units import ureg
from nomad_simulations.schema_packages.properties import ElectronicDensityOfStates
from nomad_simulations.schema_packages.variables import Energy2 as Energy


class Wannier90DosParser:
    def __init__(self, dos_file: str = ''):
        if not dos_file:
            raise ValueError('DOS `*dos.dat` file not found.')
        self.dos_parser = DataTextParser(mainfile=dos_file)

    def parse_dos(self) -> Optional[ElectronicDensityOfStates]:
        """
        Parse the `ElectronicDensityOfStates` section from the `*dos.dat` file.

        Returns:
            (Optional[ElectronicDensityOfStates]): The parsed `ElectronicDensityOfStates` property.
        """
        # TODO add spin polarized case
        data = np.transpose(self.dos_parser.data)
        sec_dos = ElectronicDensityOfStates()
        energies = Energy(points=data[0] * ureg.eV)
        sec_dos.variables.append(energies)
        sec_dos.value = data[1] / ureg.eV
        return sec_dos
