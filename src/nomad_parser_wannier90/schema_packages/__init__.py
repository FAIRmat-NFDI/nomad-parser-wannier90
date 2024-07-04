from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class Wannier90SchemaPackageEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_wannier90.schema_packages.package import m_package

        return m_package


nomad_parser_wannier90_schema = Wannier90SchemaPackageEntryPoint(
    name='Wannier90SchemaPackage',
    description='Entry point for the Wannier90 code-specific schema.',
)
