from functools import wraps
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

import re

import numpy as np
from nomad.config import config
from nomad.parsing.file_parser import Quantity, TextParser
from nomad_simulations.schema_packages.atoms_state import OrbitalsState
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.parsers:parser_entry_point'
)


def validate_atomic_cell(func):
    @wraps(func)
    def wrapper(
        self,
        position: list[float],
        atomic_cell: Optional['AtomicCell'],
        units: str,
        *args,
        **kwargs,
    ):
        if (
            atomic_cell is None
            or atomic_cell.atoms_state is None
            or len(atomic_cell.atoms_state) == 0
        ):
            print('Invalid atomic cell: either None or contains no `AtomsState`.')
            return [], []  # Return an empty tuple (or handle the error differently)
        return func(self, position, atomic_cell, units, *args, **kwargs)

    return wrapper


class WInParser(TextParser):
    def init_quantities(self):
        def str_proj_to_list(val_in):
            # To avoid inconsistent regex that can contain or not spaces
            val_n = [re.sub(r'\s.*', '', x) for x in val_in.split('\n') if x]
            return [v.strip('[]').replace(' ', '').split(':') for v in val_n]

        self._quantities = [
            Quantity(
                'energy_fermi', r'\n\rfermi_energy\s*=\s*([\d\.\-]+)', repeats=False
            ),
            Quantity(
                'projections',
                r'[bB]egin [pP]rojections([\s\S]+?)(?:[eE]nd [pP]rojections)',
                repeats=False,
                str_operation=str_proj_to_list,
            ),
        ]


class Wannier90WInParser:
    def __init__(self, win_file: str = ''):
        if not win_file:
            raise ValueError('Input `*.win` file not found.')
        self.win_parser = WInParser(mainfile=win_file)

        self._input_projection_units = {'Ang': 'angstrom', 'Bohr': 'bohr'}

        # Angular momentum [l, mr] following Wannier90 tables 3.1 and 3.2
        self._wannier_orbital_symbols_map = {
            's': ('s', ''),
            'px': ('p', 'x'),
            'py': ('p', 'y'),
            'pz': ('p', 'z'),
            'dz2': ('d', 'z^2'),
            'dxz': ('d', 'xz'),
            'dyz': ('d', 'yz'),
            'dx2-y2': ('d', 'x^2-y^2'),
            'dxy': ('d', 'xy'),
            'fz3': ('f', 'z^3'),
            'fxz2': ('f', 'xz^2'),
            'fyz2': ('f', 'yz^2'),
            'fz(x2-y2)': ('f', 'z(x^2-y^2)'),
            'fxyz': ('f', 'xyz'),
            'fx(x2-3y2)': ('f', 'x(x^2-3y^2)'),
            'fy(3x2-y2)': ('f', 'y(3x^2-y^2)'),
        }
        self._wannier_orbital_numbers_map = {
            (0, 1): ('s', ''),
            (1, 1): ('p', 'x'),
            (1, 2): ('p', 'y'),
            (1, 3): ('p', 'z'),
            (2, 1): ('d', 'z^2'),
            (2, 2): ('d', 'xz'),
            (2, 3): ('d', 'yz'),
            (2, 4): ('d', 'x^2-y^2'),
            (2, 5): ('d', 'xy'),
            (3, 1): ('f', 'z^3'),
            (3, 2): ('f', 'xz^2'),
            (3, 3): ('f', 'yz^2'),
            (3, 4): ('f', 'z(x^2-y^2)'),
            (3, 5): ('f', 'xyz'),
            (3, 6): ('f', 'x(x^2-3y^2)'),
            (3, 7): ('f', 'y(3x^2-y^2)'),
        }
        # Only angular momentum [l] (degenerate in mr)
        self._wannier_l_orbital_map = {
            0: 's',
            1: 'p',
            2: 'd',
            3: 'f',
        }

    @validate_atomic_cell
    def _convert_position(
        self, position: list[float], atomic_cell: AtomicCell, units: str
    ) -> tuple[list, list]:
        """
        Convert the atom `positions` in fractional or cartesian coordinates to the atom `chemical_symbols`.

        Args:
            position (list[float]): The position in fractional or cartesian coordinates.
            atomic_cell (AtomicCell): The `AtomicCell` section from which `positions` are extracted
            units (str): The units in which the positions are defined.

        Returns:
            tuple[list, list]: The indices and symbols at which the `position` coincides with the `AtomicCell.positions[*]`.
        """
        indices = []
        symbols = []
        for index, cell_position in enumerate(atomic_cell.positions.to(units)):
            if np.allclose(
                position,
                cell_position.magnitude,
                configuration.equal_cell_positions_tolerance,
            ):
                indices.append(index)
                symbols.append(atomic_cell.atoms_state[index].chemical_symbol)
        return indices, symbols

    def get_branch_label_and_atom_indices(
        self,
        atom: Union[str, int],
        atomic_cell: AtomicCell,
        units: str,
    ) -> tuple[str, list]:
        """
        Gets the branch label and the atom indices for the child model system.

        Args:
            atom (Union[str, int]): The atom string containing the positions information. In some older version,
            this can be an integer index pointing to the atom (which is very buggy).
            atomic_cell (AtomicCell): The `AtomicCell` section where `positions` are stored
            units (str): The units in which the positions are defined.

        Returns:
            tuple[str, list[int]]: The `branch_label` and `atom_indices` for the child model system.
        """
        # Initial check for bugs when `atom` is an integer
        if isinstance(atom, int):
            return '', [atom]

        # 3 different cases to define in `win`
        symbols: list[str] = []
        indices: list[int] = []
        # If the atom is not a chemical element, we use the `_convert_position` method resolution for it, joining the `symbols` into a long string
        if atom.startswith('f='):  # fractional coordinates
            position = [float(x) for x in atom.replace('f=', '').split(',')]
            position = np.dot(position, atomic_cell.lattice_vectors.magnitude)
            indices, symbols = self._convert_position(
                position=position, atomic_cell=atomic_cell, units=units
            )
        elif atom.startswith('c='):  # cartesian coordinates
            position = [float(x) for x in atom.replace('c=', '').split(',')]
            indices, symbols = self._convert_position(
                position=position, atomic_cell=atomic_cell, units=units
            )
        # Otherwise, if the atom chemical symbol is directly specified, we store all the `atom_indices` coinciding with this label
        else:  # atom label directly specified
            atom_indices = np.where(
                [
                    atom_state.chemical_symbol == atom
                    for atom_state in atomic_cell.atoms_state
                ]
            )[0].tolist()
            return atom, atom_indices

        branch_label = ''.join(symbols)
        return branch_label, indices

    def populate_orbitals_state(
        self,
        projection: list[str],
        model_system_child: ModelSystem,
        atomic_cell: AtomicCell,
    ) -> None:
        """
        Populate the `OrbitalsState` sections for the AtomsState relevant for the Wannier projection.

        Args:
            projection (list[Union[int, str]]): The projection information for the atom.
            model_system_child (ModelSystem): The child model system to get the `atom_indices`.
            atomic_cell (AtomicCell): The `AtomicCell` section where `positions` are stored.
            logger (BoundLogger): The logger to log messages.
        """
        # Bug when `atom` is an integer
        atom = projection[0]
        if isinstance(atom, int):
            return '', [atom]

        # Extracting orbitals information
        orbitals = projection[1].split(';')
        for atom_index in model_system_child.atom_indices:
            atom_state = atomic_cell.atoms_state[atom_index]
            for orb in orbitals:
                orbital_state = OrbitalsState()
                if orb.startswith('l='):  # using angular momentum numbers
                    lmom = int(orb.split(',mr')[0].replace('l=', '').split(',')[0])
                    if len(orb.split(',mr')) > 1:
                        mrmom = int(orb.split(',mr')[-1].replace('=', '').split(',')[0])
                        angular_momentum = self._wannier_orbital_numbers_map.get(
                            (lmom, mrmom)
                        )
                    else:
                        angular_momentum = (self._wannier_l_orbital_map.get(lmom), None)
                else:  # angular_momentum label directly specified
                    angular_momentum = self._wannier_orbital_symbols_map.get(orb)
                orbital_state.l_quantum_symbol, orbital_state.ml_quantum_symbol = (
                    angular_momentum
                )
                atom_state.orbitals_state.append(orbital_state)

    def parse_child_model_systems(
        self,
        model_system: ModelSystem,
        logger: 'BoundLogger',
    ) -> Optional[list[ModelSystem]]:
        """
        Parse the child model systems from the `*.win` file to be added as sub-sections of the parent `ModelSystem` section. We
        also store the `OrbitalsState` information of the projected atoms.

        Args:
            model_system (ModelSystem): The parent `ModelSystem` to which the child model systems will be added.
            logger (BoundLogger): The logger to log messages.

        Returns:
            (Optional[list[ModelSystem]]): The list of child model systems with the projected atoms information.
        """
        # Check if `atomic_cell` is present in `model_system``
        if model_system.cell is None or len(model_system.cell) == 0:
            logger.warning(
                'Could not extract `model_system.cell` section from mainfile `*.wout`.'
            )
            return None
        atomic_cell = model_system.cell[0]

        # Set units in case these are defined in .win
        projections = self.win_parser.get('projections', [])
        if projections:
            if not isinstance(projections, list):
                projections = [projections]
            if projections[0][0] in ['Bohr', 'Angstrom']:
                wannier90_units = self._input_projection_units[projections[0][0]]
                projections.pop(0)
            else:
                wannier90_units = 'angstrom'
            if projections[0][0] == 'random':
                return None

        # Populating AtomsGroup for projected atoms
        model_system_childs = []
        for nat in range(len(projections)):
            model_system_child = ModelSystem()
            model_system_child.type = 'active_atom'

            # atom positions information always index=0 for `projections[nat]`
            projection = projections[nat]
            atom = projection[0]
            try:
                branch_label, atom_indices = self.get_branch_label_and_atom_indices(
                    atom=atom, atomic_cell=atomic_cell, units=wannier90_units
                )
                model_system_child.branch_label = branch_label
                model_system_child.atom_indices = atom_indices
            except Exception:
                logger.warning(
                    'Error finding the atom labels for the projection from win.'
                )
                return None

            # orbital angular momentum information always index=1 for `projections[nat]`
            self.populate_orbitals_state(
                projection=projection,
                model_system_child=model_system_child,
                atomic_cell=atomic_cell,
            )
            model_system_childs.append(model_system_child)

        return model_system_childs
