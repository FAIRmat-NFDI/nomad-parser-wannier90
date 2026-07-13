from nomad.config import config
from nomad.metainfo import SchemaPackage

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


m_package.__init_metainfo__()
