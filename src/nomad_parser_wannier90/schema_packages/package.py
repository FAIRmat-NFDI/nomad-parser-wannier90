from nomad.config import config
from nomad.metainfo import SchemaPackage
from nomad.parsing.file_parser.mapping_parser import MappingAnnotationModel
from nomad_simulations.schema_packages import (
    atoms_state,
    general,
    model_method,
    model_system,
    numerical_settings,
    outputs,
    properties,
    variables,
)

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


OUT_ANNOTATION_KEY = 'out'
IN_ANNOTATION_KEY = 'in'
BAND_ANNOTATION_KEY = 'band'
HR_ANNOTATION_KEY = 'hr'
DOS_ANNOTATION_KEY = 'dos'


class Program(general.Program):

    general.Program.version.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.version')


class OrbitalsState(atoms_state.OrbitalsState):

    atoms_state.OrbitalsState.l_quantum_symbol.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.l')

    atoms_state.OrbitalsState.ml_quantum_symbol.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.m')


class AtomsState(model_system.AtomsState):

    model_system.AtomsState.chemical_symbol.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_system.AtomsState.orbitals_state.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_orbitals_state', ['.projection[1]']))


class AtomicCell(model_system.AtomicCell):

    model_system.AtomicCell.atoms_state.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.labels')

    model_system.AtomicCell.atoms_state.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_system.AtomicCell.positions.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.positions', unit='angstrom')

    model_system.AtomicCell.lattice_vectors.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_lattice_vectors', ['lattice_vectors']), unit='angstrom')

    model_system.AtomicCell.periodic_boundary_conditions.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_pbc', ['lattice_vectors']))


class ModelSystem(model_system.ModelSystem):

    model_system.AtomicCell.m_def.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_system.ModelSystem.cell.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_system.ModelSystem.model_system.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_projections', ['.projections']))

    model_system.ModelSystem.branch_label.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_branch_label_indices', ['.projection[0]', 'structure.positions', 'structure.labels','lattice_vectors']), search='label', cache=True)

    model_system.ModelSystem.atom_indices.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_branch_label_indices', ['.projection[0]', 'structure.positions', 'structure.labels','lattice_vectors']), search='indices', cache=True)


class KMesh(numerical_settings.KMesh):

    numerical_settings.KMesh.n_points.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.n_points')

    numerical_settings.KMesh.grid.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.grid')

    numerical_settings.KMesh.points.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_kpoints', ['.k_points']))


class KLinePath(numerical_settings.KLinePath):

    numerical_settings.KLinePath.high_symmetry_path_names.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.names')

    numerical_settings.KLinePath.high_symmetry_path_values.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.values')


class KSpace(numerical_settings.KSpace):

    numerical_settings.KSpace.k_mesh.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.k_mesh')

    numerical_settings.KSpace.k_line_path.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_k_line_path', ['.k_line_path']), cache=True)

class ModelMethod(model_method.ModelMethod):

    numerical_settings.KSpace.m_def.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')


class Wannier(model_method.Wannier):

    model_method.Wannier.is_maximally_localized.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('is_maximally_localized', ['.Niter'], dict(default=0)))

    model_method.Wannier.energy_window_outer.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.energy_windows.outer')

    model_method.Wannier.energy_window_inner.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.energy_windows.inner')

    model_method.Wannier.n_orbitals.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.Nwannier')


class ElectronicBandStructure(properties.ElectronicBandStructure):

    properties.ElectronicBandStructure.n_bands.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.Nwannier')

    properties.ElectronicBandStructure.value.m_annotations[BAND_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_data', ['.data']))


class WignerSeitz(variables.WignerSeitz):

    variables.WignerSeitz.n_points.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.n_ws_points')

    variables.WignerSeitz.points.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.ws_points')


class HoppingMatrix(properties.HoppingMatrix):

    properties.HoppingMatrix.n_orbitals.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='n_orbitals')

    properties.HoppingMatrix.degeneracy_factors.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.degeneracy_factors')

    # TODO shape mismatch
    # properties.HoppingMatrix.value.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.hoppings', unit='eV')

    variables.WignerSeitz.m_def.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')


class CrystalFieldSplitting(properties.CrystalFieldSplitting):

    properties.CrystalFieldSplitting.n_orbitals.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='n_orbitals')

    properties.CrystalFieldSplitting.value.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.crystal_fields', unit='eV')


class Energy2(variables.Energy2):

    variables.Energy2.points.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.energies', unit='eV')


class ElectronicDensityOfStates(properties.ElectronicDensityOfStates):

    properties.ElectronicDensityOfStates.value.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.value', unit='1/eV')

    variables.Energy2.m_def.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')


class Outputs(outputs.Outputs):

    outputs.Outputs.electronic_band_structures.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    outputs.Outputs.electronic_band_structures.m_annotations[BAND_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    outputs.Outputs.hopping_matrices.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_hoppings', ['.@'], dict(ws=True)))

    outputs.Outputs.crystal_field_splittings.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_hoppings', ['.@']))

    outputs.Outputs.electronic_dos.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_dos', ['.data']))


class Simulation(general.Simulation):

    general.Simulation.program.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.model_system.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.structure')

    general.Simulation.model_system.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_method.Wannier.m_def.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.outputs.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.outputs.m_annotations[BAND_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.outputs.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.outputs.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')


Simulation.m_def.m_annotations[OUT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')

Simulation.m_def.m_annotations[IN_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')

Simulation.m_def.m_annotations[BAND_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')

Simulation.m_def.m_annotations[HR_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')

Simulation.m_def.m_annotations[DOS_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')


m_package.__init_metainfo__()
