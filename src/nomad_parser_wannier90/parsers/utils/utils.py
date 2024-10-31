from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive

import os
from glob import glob

# from nomad.datamodel.metainfo.workflow_new import TaskReference2 as TaskReference
from nomad_simulations.schema_packages.workflow import DFTPlusTB


def get_files(pattern: str, filepath: str, stripname: str = '', deep: bool = True):
    """Get files following the `pattern` with respect to the file `stripname` (usually this
    being the mainfile of the given parser) up to / down from the `filepath` (`deep=True` going
    down, `deep=False` up)

    Args:
        pattern (str): targeted pattern to be found
        filepath (str): filepath to start the search
        stripname (str, optional): name with respect to which do the search. Defaults to ''.
        deep (bool, optional): boolean setting the path in the folders to scan (down or up). Defaults to down=True.

    Returns:
        list: List of found files.
    """
    for _ in range(10):
        filenames = glob(f'{os.path.dirname(filepath)}/{pattern}')
        pattern = os.path.join('**' if deep else '..', pattern)
        if filenames:
            break

    if len(filenames) > 1:
        # filter files that match
        suffix = os.path.basename(filepath).strip(stripname)
        matches = [f for f in filenames if suffix in f]
        filenames = matches if matches else filenames

    filenames = [f for f in filenames if os.access(f, os.F_OK)]
    return filenames


def parse_dft_plus_tb_workflow(
    dft_archive: 'EntryArchive', tb_archive: 'EntryArchive'
) -> DFTPlusTB:
    """
    Parses the DFT+TB workflow by using the DFT and TB archives.

    Args:
        dft_archive (EntryArchive): The DFT archive.
        tb_archive (EntryArchive): The TB archive.

    Returns:
        DFTPlusTB: The parsed DFT+TB workflow section.
    """
    dft_plus_tb = DFTPlusTB()

    if not dft_archive.workflow2 or not tb_archive.workflow2:
        return

    dft_task = dft_archive.workflow2
    tb_task = tb_archive.workflow2
    dft_plus_tb.tasks = [
        TaskReference(task=dft_task),
        TaskReference(task=tb_task),
    ]

    # Check if main input and output of the `DFTPlusTB` workflow exist
    # if not dft_task.m_xpath('inputs[0]') or not tb_task.m_xpath('outputs[-1]'):
    #     return dft_plus_tb
    # dft_plus_tb.inputs = [dft_task.inputs[0]]
    # dft_plus_tb.outputs = [tb_task.outputs[-1]]

    return dft_plus_tb
