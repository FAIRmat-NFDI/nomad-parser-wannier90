from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.workflow import Link, Task, Workflow
from nomad.metainfo import Quantity, Reference, SchemaPackage, SubSection
from nomad_simulations.schema_packages.model_method import BaseModelMethod
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs, SCFOutputs
from nomad_simulations.schema_packages.properties import FermiLevel

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.schema_packages:nomad_parser_wannier90_schema'
)

m_package = SchemaPackage()


class SimulationWorkflow(Workflow):
    """
    A base section used to define the workflows of a simulation with references to specific `tasks`, `inputs`, and `outputs`. The
    normalize function checks the definition of these sections and sets the name of the workflow.

    A `SimulationWorkflow` will be composed of:
        - a `method` section containing methodological parameters used specifically during the workflow,
        - a list of `inputs` with references to the `ModelSystem` or `ModelMethod` input sections,
        - a list of `outputs` with references to the `Outputs` section,
        - a list of `tasks` containing references to the activity `Simulation` used in the workflow,
    """

    method = SubSection(
        sub_section=BaseModelMethod.m_def,
        description="""Methodological parameters used during the workflow.""",
    )

    def resolve_inputs_outputs(
        self, archive: 'EntryArchive', logger: 'BoundLogger'
    ) -> None:
        """
        Resolves the `inputs` and `outputs` sections from the archive sections under `data` and stores
        them in private attributes.

        Args:
            archive (EntryArchive): The archive to resolve the sections from.
            logger (BoundLogger): The logger to log messages.
        """
        if (
            not archive.data.model_system
            or not archive.data.model_method
            or not archive.data.outputs
        ):
            logger.info(
                '`ModelSystem`, `ModelMethod` and `Outputs` required for normalization of `SimulationWorkflow`.'
            )
            return None
        self._input_systems = archive.data.model_system
        self._input_methods = archive.data.model_method
        self._outputs = archive.data.outputs

        # Resolve `inputs`
        if not self.inputs:
            self.m_add_sub_section(
                Workflow.inputs,
                Link(name='Input Model System', section=self._input_systems[0]),
            )
        # Resolve `outputs`
        if not self.outputs:
            self.m_add_sub_section(
                Workflow.outputs,
                Link(name='Output Data', section=self._outputs[-1]),
            )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Resolve the `inputs` and `outputs` from the archive
        self.resolve_inputs_outputs(archive=archive, logger=logger)

        # Storing the initial `ModelSystem`
        for link in self.inputs:
            if isinstance(link.section, ModelSystem):
                self.initial_structure = link.section
                break


class SinglePoint(SimulationWorkflow):
    """
    A `SimulationWorkflow` used to represent a single point calculation workflow. The `SinglePoint`
    workflow is the minimum workflow required to represent a simulation. The self-consistent steps of
    scf simulation are represented in the `SinglePoint` workflow.
    """

    n_scf_steps = Quantity(
        type=np.int32,
        description="""
        The number of self-consistent field (SCF) steps in the simulation.
        """,
    )

    def generate_task(self) -> Task:
        """
        Generates the `Task` section for the `SinglePoint` workflow with their `inputs` and `outputs`.

        Returns:
            Task: The generated `Task` section.
        """
        task = Task()
        if self._input_systems is not None and len(self._input_systems) > 0:
            task.m_add_sub_section(
                Task.inputs,
                Link(name='Input Model System', section=self._input_systems[0]),
            )
        if self._input_methods is not None and len(self._input_methods) > 0:
            task.m_add_sub_section(
                Task.inputs,
                Link(name='Input Model Method', section=self._input_methods[0]),
            )
        if self._outputs is not None and len(self._outputs) > 0:
            task.m_add_sub_section(
                Task.outputs,
                Link(name='Output Data', section=self._outputs[-1]),
            )
        return task

    def resolve_n_scf_steps(self) -> int:
        """
        Resolves the number of self-consistent field (SCF) steps in the simulation.

        Returns:
            int: The number of SCF steps.
        """
        for output in self.outputs:
            if not isinstance(output, SCFOutputs):
                continue
            if output.scf_steps is not None:
                return len(output.scf_steps)
        return 1

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.tasks is not None and len(self.tasks) > 1:
            logger.error('A `SinglePoint` workflow must have only one task.')
            return

        # Generate the `tasks` section if this does not exist
        if not self.tasks:
            task = self.generate_task()
            self.tasks.append(task)

        # Resolve `n_scf_steps`
        self.n_scf_steps = self.resolve_n_scf_steps()


class BeyondDFTMethod(ArchiveSection):
    """
    An abstract section used to store references to the `ModelMethod` sections of each of the
    archives defining the `tasks` and used to build the standard workflow. This section needs to be
    inherit and the method references need to be defined for each specific case.
    """

    def resolve_beyonddft_method_ref(self, task: Task) -> Optional[BaseModelMethod]:
        """
        Resolves the `ModelMethod` reference for the `task`.

        Args:
            task (Task): The task to resolve the `ModelMethod` reference from.

        Returns:
            Optional[BaseModelMethod]: The resolved `ModelMethod` reference.
        """
        for input in task.inputs:
            if input.name == 'Input Model Method':
                return input.section
        return None


class BeyondDFTWorkflow(SimulationWorkflow):
    method = SubSection(sub_section=BeyondDFTMethod.m_def)

    def resolve_all_outputs(self) -> list[Outputs]:
        """
        Resolves all the `Outputs` sections from the `tasks` in the workflow. This is useful when
        the workflow is composed of multiple tasks and the outputs need to be stored in a list
        for further manipulation, e.g., to plot multiple band structures in a DFT+TB workflow.

        Returns:
            list[Outputs]: A list of all the `Outputs` sections from the `tasks`.
        """
        all_outputs = []
        for task in self.tasks:
            all_outputs.append(task.outputs[-1])
        return all_outputs

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class DFTPlusTBMethod(BeyondDFTMethod):
    """
    Section used to reference the `DFT` and `TB` `ModelMethod` sections in each of the archives
    conforming a DFT+TB simulation workflow.
    """

    dft_method_ref = Quantity(
        type=Reference(BaseModelMethod),
        description="""Reference to the DFT `ModelMethod` section in the DFT task.""",
    )
    tb_method_ref = Quantity(
        type=Reference(BaseModelMethod),
        description="""Reference to the GW `ModelMethod` section in the TB task.""",
    )


class DFTPlusTB(BeyondDFTWorkflow):
    """
    DFT+TB workflow is composed of two tasks: the initial DFT calculation + the final TB projection. This
    workflow section is used to define the same energy reference for both the DFT and TB calculations, by
    setting it up to the DFT calculation. The structure of the workflow is:

        - `self.inputs[0]`: the initial `ModelSystem` section in the DFT entry,
        - `self.outputs[0]`: the outputs section in the TB entry,
        - `tasks[0]`:
            - `tasks[0].task` (TaskReference): the reference to the `SinglePoint` task in the DFT entry,
            - `tasks[0].inputs[0]`: the initial `ModelSystem` section in the DFT entry,
            - `tasks[0].outputs[0]`: the outputs section in the DFT entry,
        - `tasks[1]`:
            - `tasks[1].task` (TaskReference): the reference to the `SinglePoint` task in the TB entry,
            - `tasks[1].inputs[0]`: the outputs section in the DFT entry,
            - `tasks[1].outputs[0]`: the outputs section in the TB entry,
        - `method`: references to the `ModelMethod` sections in the DFT and TB entries.
    """

    def resolve_method(self) -> DFTPlusTBMethod:
        """
        Resolves the `DFT` and `TB` `ModelMethod` references for the `tasks` in the workflow by using the
        `resolve_beyonddft_method_ref` method from the `BeyondDFTMethod` section.

        Returns:
            DFTPlusTBMethod: The resolved `DFTPlusTBMethod` section.
        """
        method = DFTPlusTBMethod()

        # DFT method reference
        dft_method = method.resolve_beyonddft_method_ref(task=self.tasks[0].task)
        if dft_method is not None:
            method.dft_method_ref = dft_method

        # TB method reference
        tb_method = method.resolve_beyonddft_method_ref(task=self.tasks[1].task)
        if tb_method is not None:
            method.tb_method_ref = tb_method

        return method

    def link_tasks(self) -> None:
        """
        Links the `outputs` of the DFT task with the `inputs` of the TB task.
        """
        dft_task = self.tasks[0]
        dft_task.inputs = [
            Link(
                name='Input Model System',
                section=self.inputs[0],
            )
        ]
        dft_task.outputs = [
            Link(
                name='Output DFT Data',
                section=dft_task.outputs[-1],
            )
        ]

        tb_task = self.tasks[1]
        tb_task.inputs = [
            Link(
                name='Output DFT Data',
                section=dft_task.outputs[-1],
            ),
        ]
        tb_task.outputs = [
            Link(
                name='Output TB Data',
                section=tb_task.outputs[-1],
            )
        ]

    def overwrite_fermi_level(self) -> None:
        """
        Overwrites the Fermi level in the TB calculation with the Fermi level from the DFT calculation.
        """
        dft_output = self.tasks[0].outputs[-1]
        if not dft_output.fermi_levels:
            return None
        fermi_level = dft_output.fermi_levels[-1]

        tb_output = self.tasks[1].outputs[-1]
        tb_output.fermi_levels.append(FermiLevel(value=fermi_level.value))

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Initial check for the number of tasks
        if len(self.tasks) != 2:
            logger.error('A `DFTPlusTB` workflow must have two tasks.')
            return

        # Check if tasks are `SinglePoint`
        for task in self.tasks:
            if task.m_def.name != 'SinglePoint':
                logger.error(
                    'A `DFTPlusTB` workflow must have two `SinglePoint` tasks.'
                )
                return

        # Define names of the workflow and `tasks`
        self.name = 'DFT+TB'
        self.tasks[0].name = 'DFT SinglePoint'
        self.tasks[1].name = 'TB SinglePoint'

        # Resolve method refs for each task and store under `method`
        self.method = self.resolve_method()

        # Link the tasks
        self.link_tasks()

        # Overwrite the Fermi level in the TB calculation
        self.overwrite_fermi_level()


m_package.__init_metainfo__()
