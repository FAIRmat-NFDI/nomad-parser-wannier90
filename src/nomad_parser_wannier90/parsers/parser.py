import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from nomad.config import config

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.parsing.file_parser import Quantity, TextParser
from nomad.units import ureg
from nomad_simulations.schema_packages.atoms_state import AtomsState

# New schema
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod, Wannier
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
from nomad_simulations.schema_packages.numerical_settings import (
    KLinePath,
    KMesh,
    KSpace,
)
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.workflow import SinglePoint

# from nomad_simulations.schema_packages.utils import check_simulation_cell
from nomad_parser_wannier90.parsers.band_parser import Wannier90BandParser
from nomad_parser_wannier90.parsers.dos_parser import Wannier90DosParser
from nomad_parser_wannier90.parsers.hr_parser import Wannier90HrParser
from nomad_parser_wannier90.parsers.utils import get_files, parse_dft_plus_tb_workflow
from nomad_parser_wannier90.parsers.win_parser import Wannier90WInParser

re_n = r'[\n\r]'

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.parsers:parser_entry_point'
)


class WOutParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        kmesh_quantities = [
            Quantity('n_points', r'Total points[\s=]*(\d+)', dtype=int, repeats=False),
            Quantity(
                'grid', r'Grid size *\= *(\d+) *x *(\d+) *x *(\d+)', repeats=False
            ),
            Quantity('k_points', r'\|[\s\d]*(-*\d.[^\|]+)', repeats=True, dtype=float),
        ]

        klinepath_quantities = [
            Quantity(
                'high_symm_name',
                r'\| *From\: *([a-zA-Z]+) [\d\.\-\s]*To\: *([a-zA-Z]+)',
                repeats=True,
            ),
            Quantity(
                'high_symm_value',
                r'\| *From\: *[a-zA-Z]* *([\d\.\-\s]+)To\: *[a-zA-Z]* *([\d\.\-\s]+)\|',
                repeats=True,
            ),
        ]

        disentangle_quantities = [
            Quantity(
                'outer',
                r'\|\s*Outer:\s*([-\d.]+)\s*\w*\s*([-\d.]+)\s*\((?P<__unit>\w+)\)',
                dtype=float,
                repeats=False,
            ),
            Quantity(
                'inner',
                r'\|\s*Inner:\s*([-\d.]+)\s*\w*\s*([-\d.]+)\s*\((?P<__unit>\w+)\)',
                dtype=float,
                repeats=False,
            ),
        ]

        structure_quantities = [
            Quantity('labels', r'\|\s*([A-Z][a-z]*)', repeats=True),
            Quantity(
                'positions',
                r'\|\s*([\-\d\.]+)\s*([\-\d\.]+)\s*([\-\d\.]+)',
                repeats=True,
                dtype=float,
            ),
        ]

        self._quantities = [
            # Program quantities
            Quantity(
                Program.version, r'\s*\|\s*Release\:\s*([\d\.]+)\s*', repeats=False
            ),
            # System quantities
            Quantity('lattice_vectors', r'\s*a_\d\s*([\d\-\s\.]+)', repeats=True),
            Quantity(
                'reciprocal_lattice_vectors', r'\s*b_\d\s*([\d\-\s\.]+)', repeats=True
            ),
            Quantity(
                'structure',
                rf'(\s*Fractional Coordinate[\s\S]+?)(?:{re_n}\s*(PROJECTIONS|K-POINT GRID))',
                repeats=False,
                sub_parser=TextParser(quantities=structure_quantities),
            ),
            # Method quantities
            Quantity(
                'k_mesh',
                r'\s*(K-POINT GRID[\s\S]+?)(?:-\s*MAIN)',
                repeats=False,
                sub_parser=TextParser(quantities=kmesh_quantities),
            ),
            Quantity(
                'k_line_path',
                r'\s*(K-space path sections\:[\s\S]+?)(?:\*-------)',
                repeats=False,
                sub_parser=TextParser(quantities=klinepath_quantities),
            ),
            Quantity(
                'Nwannier',
                r'\|\s*Number of Wannier Functions\s*\:\s*(\d+)',
                repeats=False,
            ),
            Quantity(
                'Nband',
                r'\|\s*Number of input Bloch states\s*\:\s*(\d+)',
                repeats=False,
            ),
            Quantity(
                'Niter', r'\|\s*Total number of iterations\s*\:\s*(\d+)', repeats=False
            ),
            Quantity(
                'conv_tol',
                r'\|\s*Convergence tolerence\s*\:\s*([\d.eE-]+)',
                repeats=False,
            ),
            Quantity(
                'energy_windows',
                r'(\|\s*Energy\s*Windows\s*\|[\s\S]+?)(?:Number of target bands to extract:)',
                repeats=False,
                sub_parser=TextParser(quantities=disentangle_quantities),
            ),
            # Band related quantities
            Quantity(
                'n_k_segments',
                r'\|\s*Number of K-path sections\s*\:\s*(\d+)',
                repeats=False,
            ),
            Quantity(
                'div_first_k_segment',
                r'\|\s*Divisions along first K-path section\s*\:\s*(\d+)',
                repeats=False,
            ),
            Quantity(
                'band_segments_points',
                r'\|\s*From\:\s*\w+([\d\s\-\.]+)To\:\s*\w+([\d\s\-\.]+)',
                repeats=True,
            ),
        ]


class Wannier90Parser:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.wout_parser = WOutParser()

        self._dft_codes = [
            'quantumespresso',
            'abinit',
            'vasp',
            'siesta',
            'wien2k',
            'fleur',
            'openmx',
            'gpaw',
        ]

        self._input_projection_mapping = {
            'Nwannier': 'n_orbitals',
            'Nband': 'n_bloch_bands',
        }

        self._child_archives = {}

    def init_parser(self, logger: 'BoundLogger') -> None:
        """
        Initialize the `WOutParser` with the mainfile and logger.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        self.wout_parser.mainfile = self.mainfile
        self.wout_parser.logger = logger

    def parse_atoms_state(self, labels: Optional[list[str]]) -> list[AtomsState]:
        """
        Parse the `AtomsState` from the labels by storing them as the `chemical_symbols`.

        Args:
            labels (Optional[list[str]]): List of chemical element labels.

        Returns:
            list[AtomsState]: List of `AtomsState` sections.
        """
        if labels is None:
            return []
        atoms_state = []
        for label in labels:
            atoms_state.append(AtomsState(chemical_symbol=label))
        return atoms_state

    def parse_atomic_cell(self) -> AtomicCell:
        """
        Parse the `AtomicCell` from the `lattice_vectors` and `structure` regex quantities in `WOutParser`.

        Returns:
            AtomicCell: The parsed `AtomicCell` section.
        """
        atomic_cell = AtomicCell()

        # Parsing `lattice_vectors`
        if self.wout_parser.get('lattice_vectors', []):
            lattice_vectors = np.vstack(
                self.wout_parser.get('lattice_vectors', [])[-3:]
            )
            atomic_cell.lattice_vectors = lattice_vectors * ureg.angstrom
        # and `periodic_boundary_conditions`
        pbc = (
            [True, True, True] if lattice_vectors is not None else [False, False, False]
        )
        atomic_cell.periodic_boundary_conditions = pbc

        # Parsing `atoms_state` from `structure`
        labels = self.wout_parser.get('structure', {}).get('labels')
        if labels is not None:
            atoms_state = self.parse_atoms_state(labels=labels)
            atomic_cell.atoms_state = atoms_state
        # and parsing `positions`
        if self.wout_parser.get('structure', {}).get('positions') is not None:
            atomic_cell.positions = (
                self.wout_parser.get('structure', {}).get('positions') * ureg.angstrom
            )
        return atomic_cell

    def parse_model_system(self, logger: 'BoundLogger') -> Optional[ModelSystem]:
        """
        Parse the `ModelSystem` with the `AtomicCell` information. If the `structure` is not recognized in `WOutParser`, then return `None`.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[ModelSystem]: The parsed `ModelSystem` section.
        """
        model_system = ModelSystem()
        model_system.is_representative = True

        # If the `structure` is not parsed, return None
        if self.wout_parser.get('structure') is None:
            logger.error('Error parsing the structure from .wout')
            return None

        atomic_cell = self.parse_atomic_cell()
        model_system.cell.append(atomic_cell)
        return model_system

    def parse_wannier(self) -> Wannier:
        """
        Parse the `Wannier` section from the `WOutParser` quantities.

        Returns:
            Wannier: The parsed `Wannier` section.
        """
        model_wannier = Wannier()
        for key in self._input_projection_mapping.keys():
            setattr(
                model_wannier,
                self._input_projection_mapping[key],
                self.wout_parser.get(key),
            )
        if self.wout_parser.get('Niter'):
            model_wannier.is_maximally_localized = self.wout_parser.get('Niter', 0) > 1
        model_wannier.energy_window_outer = self.wout_parser.get(
            'energy_windows', {}
        ).get('outer')
        model_wannier.energy_window_inner = self.wout_parser.get(
            'energy_windows', {}
        ).get('inner')
        return model_wannier

    def parse_k_mesh(self) -> Optional[KMesh]:
        """
        Parse the `KMesh` section from the `WOutParser` quantities.

        Returns:
            Optional[KMesh]: The parsed `KMesh` section.
        """
        sec_k_mesh = None
        k_mesh = self.wout_parser.get('k_mesh')
        if k_mesh is None:
            return sec_k_mesh
        sec_k_mesh = KMesh()
        sec_k_mesh.n_points = k_mesh.get('n_points')
        sec_k_mesh.grid = k_mesh.get('grid', [])
        if k_mesh.get('k_points') is not None:
            sec_k_mesh.points = np.complex128(k_mesh.k_points[::2])
        return sec_k_mesh

    def parse_k_line_path(self) -> Optional[KLinePath]:
        """
        Parse the `KLinePath` section from the `WOutParser` quantities.

        Returns:
            Optional[KLinePath]: The parsed `KLinePath` section.
        """
        sec_k_line_path = None
        k_line_path = self.wout_parser.get('k_line_path')
        if k_line_path is None:
            return sec_k_line_path

        # Store the list of high symmetry names and values for the section `KLinePath`
        high_symm_names = k_line_path.get('high_symm_name')
        high_symm_values = [
            np.reshape(val, (2, 3)) for val in k_line_path.get('high_symm_value')
        ]
        # Start with the first element of the first pair
        names = [high_symm_names[0][0]]
        values = [high_symm_values[0][0]]
        for i, pair in enumerate(high_symm_names):
            # Add the second element if it's not the last one in the list
            if pair[1] != names[-1]:
                names.append(pair[1])
                values.append(high_symm_values[i][1])
        sec_k_line_path = KLinePath(
            high_symmetry_path_names=names, high_symmetry_path_values=values
        )  # `points` are extracted in the `Wannier90BandParser` using the `KLinePath.resolve_points` method
        return sec_k_line_path

    def parse_model_method(self) -> ModelMethod:
        """
        Parse the `ModelWannier(ModelMethod)` section from the `WOutParser` quantities.

        Returns:
            ModelMethod: The parsed `ModelWannier(ModelMethod)` section.
        """
        # `ModelMethod` section
        model_wannier = self.parse_wannier()

        # `NumericalSettings` sections
        k_mesh = self.parse_k_mesh()
        if k_mesh is not None:
            k_space = KSpace(k_mesh=[k_mesh])
            model_wannier.numerical_settings.append(k_space)

        k_line_path = self.parse_k_line_path()
        if k_line_path is not None:
            if k_space is None:
                k_space = KSpace()
                model_wannier.numerical_settings.append(k_space)
            k_space.k_line_path = k_line_path

        return model_wannier

    def parse_outputs(self, simulation: Simulation, logger: 'BoundLogger') -> Outputs:
        outputs = Outputs()
        if simulation.model_system is not None:
            outputs.model_system_ref = simulation.model_system[-1]
        if simulation.model_method is not None:
            outputs.model_method_ref = simulation.model_method[-1]

        # Parse hoppings
        hr_files = get_files(
            pattern='*hr.dat', filepath=self.mainfile, stripname=self.basename
        )
        if len(hr_files) > 1:
            logger.info('Multiple `*hr.dat` files found.')
        # contains information about `n_orbitals`
        wannier_method = simulation.m_xpath('model_method[-1]', dict=False)
        for hr_file in hr_files:
            hopping_matrix, crystal_field_splitting = Wannier90HrParser(
                hr_file=hr_file
            ).parse_hoppings(wannier_method=wannier_method, logger=logger)
            if hopping_matrix is not None:
                outputs.hopping_matrices.append(hopping_matrix)
            if crystal_field_splitting is not None:
                outputs.crystal_field_splittings.append(crystal_field_splitting)

        # Parse DOS
        dos_files = get_files(
            pattern='*dos.dat', filepath=self.mainfile, stripname=self.basename
        )
        if len(dos_files) > 1:
            logger.info('Multiple `*dos.dat` files found.')
        for dos_file in dos_files:
            electronic_dos = Wannier90DosParser(dos_file=dos_file).parse_dos()
            if electronic_dos is not None:
                outputs.electronic_dos.append(electronic_dos)

        # Parse BandStructure
        band_files = get_files(
            pattern='*band.dat', filepath=self.mainfile, stripname=self.basename
        )
        # contains information about `k_line_path`
        k_space = simulation.m_xpath(
            'model_method[-1].numerical_settings[-1]', dict=False
        )
        # getting the list of `ModelSystem` to extract the reciprocal_lattice_vectors
        model_systems = simulation.m_xpath('model_system', dict=False)
        if len(band_files) > 1:
            logger.info('Multiple `*band.dat` files found.')
        for band_file in band_files:
            band_structure = Wannier90BandParser(
                band_file=band_file
            ).parse_band_structure(
                k_space=k_space,
                wannier_method=wannier_method,
                model_systems=model_systems,
                logger=logger,
            )
            if band_structure is not None:
                outputs.electronic_band_structures.append(band_structure)

        return outputs

    def workflow_dft_files(self, **kwargs) -> list[str]:
        """
        Check if in the upload of the Wannier90 mainfile, there are the corresponding DFT files to create the
        DFT+TB workflow, and returns them if so. Implemented only for VASP DFT files.

        Returns:
            list[str]: Returns a list containing the DFT files if they are present, otherwise an empty list.
        """
        # Wannier90 file
        wannier90_mainfile = kwargs.get('filename')
        wannier90_basename = os.path.basename(wannier90_mainfile)

        # DFT files
        dft_files = []

        # VASP DFT files
        vasprun_files = get_files(
            pattern='*vasprun.xml',
            filepath=wannier90_mainfile,
            stripname=wannier90_basename,
            deep=False,
        )
        outcar_files = get_files(
            pattern='*OUTCAR',
            filepath=wannier90_mainfile,
            stripname=wannier90_basename,
            deep=False,
        )
        if not vasprun_files:
            dft_files = outcar_files
        else:
            dft_files = vasprun_files

        # TODO extend to other DFT codes

        return dft_files

    def get_mainfile_keys(self, **kwargs):
        """
        Generates extra `child_archives` to create the DFT+TB workflow if the conditions in `workflow_dft_files` are met.
        """
        dft_files = self.workflow_dft_files(**kwargs)
        if not dft_files:
            return True
        return ['DFTPlusTB_workflow']

    def parse(
        self, filepath: str, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        self.mainfile = filepath
        self.maindir = os.path.dirname(self.mainfile)
        self.basename = os.path.basename(self.mainfile)
        self.archive = archive

        self.init_parser(logger=logger)

        # Adding Simulation to data
        simulation = Simulation()
        simulation.program = Program(
            name='Wannier90',
            version=self.wout_parser.get('version', ''),
        )
        archive.data = simulation

        # `ModelSystem` parsing
        model_system = self.parse_model_system(logger=logger)
        if model_system is not None:
            simulation.model_system.append(model_system)

            # Child `ModelSystem` and `OrbitalsState` parsing
            win_files = get_files(
                pattern='*.win', filepath=self.mainfile, stripname=self.basename
            )
            if len(win_files) > 1:
                logger.warning(
                    'Multiple `*.win` files found. We will parse the first one.'
                )
            if win_files is not None:
                child_model_systems = Wannier90WInParser(
                    win_file=win_files[0]
                ).parse_child_model_systems(model_system=model_system, logger=logger)
                model_system.model_system = child_model_systems

        # `ModelWannier(ModelMethod)` parsing
        model_method = self.parse_model_method()
        simulation.model_method.append(model_method)

        # `Outputs` parsing
        outputs = self.parse_outputs(simulation=simulation, logger=logger)
        simulation.outputs.append(outputs)

        # Workflow section
        workflow = SinglePoint()
        workflow.normalize(archive=archive, logger=logger)
        archive.workflow2 = workflow

        # TODO extend to handle DFT+TB workflows using `self._dft_codes`
        # Checking if other mainfiles are present, if the closest is a DFT code, tries to create the
        # DFT+TB workflow and link it with the corresponding Wannier90 entry
        dft_files = self.workflow_dft_files(filename=self.mainfile)
        if len(dft_files) == 1:
            dft_path = dft_files[-1].split('raw/')[-1]
            filepath_stripped = self.mainfile.split('raw/')[-1]
            try:
                # For automatic workflows
                from nomad.app.v1.models import MetadataRequired
                from nomad.search import search

                upload_id = self.archive.metadata.upload_id
                search_ids = search(
                    owner='visible',
                    user_id=self.archive.metadata.main_author.user_id,
                    query={'upload_id': upload_id},
                    required=MetadataRequired(include=['entry_id', 'mainfile']),
                ).data
                metadata = [[sid['entry_id'], sid['mainfile']] for sid in search_ids]
                for entry_id, mainfile in metadata:
                    if (
                        mainfile == filepath_stripped
                    ):  # we skipped the current parsed mainfile
                        continue
                    entry_archive = self.archive.m_context.load_archive(
                        entry_id, upload_id, None
                    )
                    if dft_path == mainfile:
                        dft_archive = entry_archive

                        # ! commented out for now, until VASP parser is ready
                        # # check if the simulation cell is the same
                        # dft_cell = dft_archive.m_xpath(
                        #     'data.model_system[-1].cell[0]'
                        # )
                        # tb_cell = self.archive.m_xpath(
                        #     'data.model_system[-1].cell[0]'
                        # )
                        # if dft_cell is not None and tb_cell is not None:
                        #     if dft_cell != tb_cell:
                        #         logger.warning(
                        #             'The DFT and TB cells do not coincide. We might be connecting wrongly the DFT and TB tasks.'
                        #         )
                        # else:
                        #     logger.warning(
                        #         'Could not resolve the DFT and TB cells.'
                        #     )
                        #     return

                        # Parse the workflow information
                        dft_plus_tb_archive = self._child_archives.get(
                            'DFTPlusTB_workflow'
                        )
                        dft_plus_tb = parse_dft_plus_tb_workflow(
                            dft_archive=dft_archive, tb_archive=self.archive
                        )
                        dft_plus_tb_archive.workflow2 = dft_plus_tb
                        break
            except Exception:
                logger.warning('Could not resolve the DFT+TB workflow for Wannier90.')
