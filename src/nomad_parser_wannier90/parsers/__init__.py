from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class SimulationParserEntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )
    level: int = Field(
        0,
        description="""
        Order of execution of parser with respect to other parsers.
    """,
    )

    def load(self):
        from nomad.parsing import MatchingParserInterface

        return MatchingParserInterface(**self.dict())


nomad_parser_wannier90_plugin = SimulationParserEntryPoint(
    name='parsers/wannier90',
    aliases=['parsers/wannier90'],
    description='Entry point for the Wannier90 parser.',
    python_package='nomad_parser_wannier90.parsers',
    parser_class_name='nomad_parser_wannier90.parsers.parser.Wannier90Parser',
    # parser_as_interface=False,  # in order to use `child_archives` and auto workflows
    level=1,
    mainfile_contents_re=r'\|\s*WANNIER90\s*\|',
)
