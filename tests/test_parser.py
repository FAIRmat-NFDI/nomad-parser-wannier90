import os
from typing import Optional

import numpy as np
import pytest
from nomad.datamodel import EntryArchive

from nomad_parser_wannier90.parsers.parser import Wannier90Parser

from . import logger


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return Wannier90Parser()


class Wannier90ParserTest:
    @pytest.mark.parametrize(
        'labels, result',
        [
            (None, []),
            ([], []),
            (
                ['La', 'La', 'Cu', 'O', 'O', 'O', 'O'],
                ['La', 'La', 'Cu', 'O', 'O', 'O', 'O'],
            ),
        ],
    )
    def test_parse_atoms_state(self, parser, labels: Optional[list[str]], result: list):
        """Test the `parse_atoms_state` method."""
        atoms_states = parser.parse_atoms_state(labels=labels)
        if len(atoms_states) == 0:
            assert atoms_states == result
        else:
            for index, atom in atoms_states:
                assert atom.chemical_symbol == result[index]


def test_single_point_La2CuO4(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'lco_mlwf', 'lco.wout'),
        archive,
        logger,
    )
    simulation = archive.data

    # Program
    assert simulation.program.name == 'Wannier90'
    assert simulation.program.version == '3.1.0'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    #   Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]
    assert np.isclose(
        atomic_cell.positions.to('angstrom').magnitude,
        np.array(
            [
                [-0.0, -0.0, 4.77028],
                [1.90914, 1.90914, 1.83281],
                [-0.0, -0.0, -0.0],
                [-0.0, 1.90914, 0.0],
                [1.90914, -0.0, 0.0],
                [-0.0, -0.0, 2.45222],
                [1.90914, 1.90914, 4.15087],
            ]
        ),
    ).all()
    assert np.isclose(
        atomic_cell.lattice_vectors.to('angstrom').magnitude,
        np.array(
            [
                [-1.909145, 1.909145, 6.603098],
                [1.909145, -1.909145, 6.603098],
                [1.909145, 1.909145, -6.603098],
            ]
        ),
    ).all()
    assert atomic_cell.periodic_boundary_conditions == [True, True, True]
    #       AtomsState
    assert len(atomic_cell.atoms_state) == 7
    for index, symbol in enumerate(['La', 'La', 'Cu', 'O', 'O', 'O', 'O']):
        assert atomic_cell.atoms_state[index].chemical_symbol == symbol
    #   SubSystem
    assert model_system.model_system[0].type == 'active_atom'
    assert model_system.model_system[0].branch_label == 'Cu'
    assert (model_system.model_system[0].atom_indices == [2]).all()

    # ModelMethod
    assert len(simulation.model_method) == 1
    assert simulation.model_method[0].m_def.name == 'Wannier'
    wannier = simulation.model_method[0]
    assert wannier.is_maximally_localized
    assert wannier.n_orbitals == 1
    assert wannier.n_bloch_bands == 5
    assert np.isclose(wannier.energy_window_inner.magnitude, [12.3, 16.0]).all()
    assert np.isclose(wannier.energy_window_outer.magnitude, [10.0, 16.0]).all()
    #   NumericalSettings
    assert len(wannier.numerical_settings) == 1
    assert wannier.numerical_settings[0].m_def.name == 'KSpace'
    k_space = wannier.numerical_settings[0]
    #       KMesh
    assert len(k_space.k_mesh) == 1
    assert k_space.k_mesh[0].n_points == 343
    assert np.isclose(k_space.k_mesh[0].grid, [7, 7, 7]).all()
    #       KLinePath
    assert k_space.k_line_path.n_line_points == 371
    assert k_space.k_line_path.high_symmetry_path_names == ['G', 'N', 'X', 'G', 'M']

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == wannier
    #   Properties
    for property_name in [
        'crystal_field_splittings',
        'hopping_matrices',
        'electronic_dos',
        'electronic_band_structures',
    ]:
        assert output.m_xpath(property_name, dict=False) is not None
        assert len(output.m_xpath(property_name, dict=False)) == 1
    #       CrystalFieldSplitting
    assert output.crystal_field_splittings[0].value[0].to('eV').magnitude == approx(
        12.895622
    )
    #       HoppingMatrix
    assert len(output.hopping_matrices[0].variables) == 1
    assert output.hopping_matrices[0].variables[0].m_def.name == 'WignerSeitz'
    assert output.hopping_matrices[0].variables[0].n_points == 397
    assert output.hopping_matrices[0].value.shape == (397, 1, 1)
    assert output.hopping_matrices[0].value[4, 0, 0].to('eV').magnitude == approx(
        5e-06 + 0j
    )
    #       ElectronicDOS
    assert len(output.electronic_dos[0].variables) == 1
    assert output.electronic_dos[0].variables[0].m_def.name == 'Energy2'
    assert len(output.electronic_dos[0].variables[0].points) == 692
    assert output.electronic_dos[0].value.shape == (692,)
    assert output.electronic_dos[0].value[222].magnitude == approx(2.506209437080082e18)
    #       ElectronicBandStructure
    assert len(output.electronic_band_structures[0].variables) == 1
    assert output.electronic_band_structures[0].variables[0].m_def.name == 'KLinePath'
    assert len(output.electronic_band_structures[0].variables[0].points) == 371
    assert output.electronic_band_structures[0].value.shape == (371, 1)
    assert output.electronic_band_structures[0].value[123, 0].to(
        'eV'
    ).magnitude == approx(12.870674)


def test_minimal_Pt_Vikrant(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'minimal_Pt_Vikrant', 'wannier90.wout'),
        archive,
        logger,
    )
    simulation = archive.data

    # Program
    assert simulation.program.name == 'Wannier90'
    assert simulation.program.version == '3.1.0'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    #   Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]
    assert np.isclose(
        atomic_cell.positions.to('angstrom').magnitude,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.972, 1.972],
                [1.972, 0.0, 1.972],
                [1.972, 1.972, 0.0],
            ]
        ),
    ).all()
    assert np.isclose(
        atomic_cell.lattice_vectors.to('angstrom').magnitude,
        np.array([[3.944, 0.0, 0.0], [0.0, 3.944, 0.0], [0.0, 0.0, 3.944]]),
    ).all()
    assert atomic_cell.periodic_boundary_conditions == [True, True, True]
    #       AtomsState
    assert len(atomic_cell.atoms_state) == 4
    for index, symbol in enumerate(['Pt', 'Pt', 'Pt', 'Pt']):
        assert atomic_cell.atoms_state[index].chemical_symbol == symbol
    #   SubSystems
    assert len(model_system.model_system) == 4
    for i, subsystem in enumerate(model_system.model_system):
        assert subsystem.type == 'active_atom'
        assert subsystem.branch_label == 'Pt'
        assert (subsystem.atom_indices == [i]).all()

    # ModelMethod
    assert len(simulation.model_method) == 1
    assert simulation.model_method[0].m_def.name == 'Wannier'
    wannier = simulation.model_method[0]
    assert wannier.is_maximally_localized
    # 's', 'p', 'd' deg orbitals used for all 4 Pt -> 4 * 3 = 12 orbitals * 3 Pt atoms per unit cell = 36
    assert wannier.n_orbitals == 36
    assert wannier.n_bloch_bands == 240
    assert np.isclose(wannier.energy_window_inner.magnitude, [8.80301, 11.80301]).all()
    assert np.isclose(wannier.energy_window_outer.magnitude, [0.32315, 36.58885]).all()
    #   NumericalSettings
    assert len(wannier.numerical_settings) == 1
    assert wannier.numerical_settings[0].m_def.name == 'KSpace'
    k_space = wannier.numerical_settings[0]
    #       KMesh
    assert len(k_space.k_mesh) == 1
    assert k_space.k_mesh[0].n_points == 512
    assert np.isclose(k_space.k_mesh[0].grid, [8, 8, 8]).all()
    #       KLinePath
    assert not k_space.m_xpath('k_line_path')

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == wannier
    #   Properties
    for property_name in ['crystal_field_splittings', 'hopping_matrices']:
        assert output.m_xpath(property_name, dict=False) is not None
        assert len(output.m_xpath(property_name, dict=False)) == 1
    #       CrystalFieldSplitting
    assert output.crystal_field_splittings[0].value[0].to('eV').magnitude == approx(
        11.966103
    )
    #       HoppingMatrix
    assert len(output.hopping_matrices[0].variables) == 1
    assert output.hopping_matrices[0].variables[0].m_def.name == 'WignerSeitz'
    assert output.hopping_matrices[0].variables[0].n_points == 729
    assert output.hopping_matrices[0].value.shape == (729, 36, 36)
    assert output.hopping_matrices[0].value[4, 0, 0].to('eV').magnitude == approx(
        0.000224 + 0j
    )
    #       ElectronicDOS
    assert not output.m_xpath('electronic_dos')
    #       ElectronicBandStructure
    assert not output.m_xpath('electronic_band_structures')
