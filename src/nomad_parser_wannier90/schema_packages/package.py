from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.metainfo import SchemaPackage
from nomad_simulations.schema_packages.general import Simulation

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.schema_packages:nomad_parser_wannier90_schema'
)

m_package = SchemaPackage()


# not necessary: just for demonstration purposes
class Wannier90Simulation(Simulation):
    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


m_package.__init_metainfo__()
