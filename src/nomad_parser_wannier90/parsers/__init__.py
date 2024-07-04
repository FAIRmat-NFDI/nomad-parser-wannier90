from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class Wannier90ParserEntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_wannier90.parsers.parser import Wannier90Parser

        return Wannier90Parser(**self.dict())


nomad_parser_wannier90_plugin = Wannier90ParserEntryPoint(
    name='Wannier90ParserEntryPoint',
    description='Entry point for the Wannier90 parser.',
    level=1,
    mainfile_contents_re=r'\|\s*WANNIER90\s*\|',
)
