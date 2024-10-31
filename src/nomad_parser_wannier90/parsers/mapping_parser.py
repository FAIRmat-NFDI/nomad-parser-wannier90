import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from nomad.config import config

if TYPE_CHECKING:
    from nomad.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.parsing.file_parser.mapping_parser import MetainfoParser, TextParser
from nomad.parsing.file_parser.text_parser import DataTextParser
from nomad_simulations.schema_packages.workflow import SinglePoint

from nomad_parser_wannier90.schema_packages.package import (
    BAND_ANNOTATION_KEY,
    DOS_ANNOTATION_KEY,
    HR_ANNOTATION_KEY,
    IN_ANNOTATION_KEY,
    OUT_ANNOTATION_KEY,
    Simulation,
)

from .hr_parser import HrParser
from .parser import WOutParser
from .utils import get_files, parse_dft_plus_tb_workflow
from .win_parser import WInParser

re_n = r'[\n\r]'

configuration = config.get_plugin_entry_point(
    'nomad_parser_wannier90.parsers:parser_entry_point'
)


class WHrTextParser(TextParser):
    def get_hoppings(self, source: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        degeneracy_factors = source.get('degeneracy_factors')[2:]
        full_hoppings = source.get('hoppings', [])
        n_wigner_seitz_points = source.get('degeneracy_factors')[1]
        n_orbitals = source.get('n_orbitals')

        hops = np.reshape(
            full_hoppings,
            (n_wigner_seitz_points, n_orbitals, n_orbitals, 7),
        )

        # storing the crystal field splitting values
        ws0 = int((n_wigner_seitz_points - 1) / 2)
        crystal_fields = [
            hops[ws0, i, i, 5] for i in range(n_orbitals)
        ]  # only real elements

        # delete repeated points for different orbitals
        ws_points = hops[:, :, :, :3]
        ws_points = np.unique(ws_points.reshape(-1, 3), axis=0)

        # passing hoppings
        hoppings = hops[:, :, :, -2] + 1j * hops[:, :, :, -1]
        result = dict(
            degeneracy_factors=degeneracy_factors,
            hoppings=hoppings,
            crystal_fields=crystal_fields,
        )
        if kwargs.get('ws'):
            result.update(dict(ws_points=ws_points, n_ws_points=n_wigner_seitz_points))

        return result


class WDosTextParser(TextParser):
    def get_dos(self, source: np.ndarray) -> Dict[str, Any]:
        data = np.transpose(source)
        return dict(energies=data[0], value=data[1])


class WBandTextParser(TextParser):
    def get_data(self, data: np.ndarray) -> np.ndarray:
        return np.transpose(data)[1:].transpose()


class WOutTextParser(TextParser):
    def get_lattice_vectors(self, vectors: List[Any]) -> np.ndarray:
        return np.vstack(vectors[-3:])

    def get_pbc(self, vectors: List[Any]) -> List[bool]:
        return [vectors is not None] * 3

    def is_maximally_localized(self, niter: int, default=0) -> bool:
        return (niter or default) > 1

    def get_kpoints(self, points: np.ndarray) -> np.ndarray:
        return np.complex128(points[::2])

    def get_k_line_path(self, k_line_path: Dict[str, Any]):
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
        return dict(names=names, values=values)


class WInTextParser(TextParser):
    # TODO these should be defined in common utils
    _l_symbols = ['s', 'p', 'd', 'f']
    _m_symbols = [
        None,
        'x',
        'y',
        'z',
        'z^2',
        'xz',
        'yz',
        'x^2-y^2',
        'xy',
        'z^3',
        'xz^2',
        'yz^2',
        'z(x^2-y^2)',
        'xyz',
        'x(x^2-3y^2)',
        'y(3x^2-y^2)',
    ]
    _wannier_symbols = [
        's',
        'px',
        'py',
        'pz',
        'dz2',
        'dxz',
        'dyz',
        'dx2-y2',
        'dxy',
        'fz3',
        'fxz2',
        'fyz2',
        'fz(x2-y2)',
        'fxyz',
        'fx(x2-3y2)',
        'fy(3x2-y2)',
    ]

    def get_projections(self, source: List[Any]) -> List[Dict[str, Any]]:
        return [dict(projection=val) for val in source]

    def get_branch_label_indices(
        self,
        atom: Any,
        positions: List[np.ndarray],
        labels: List[str],
        lattice_vectors: List[np.ndarray],
    ) -> Any:
        symbols, indices = [], []
        if atom is None:
            return None

        elif isinstance(atom, int):
            indices = [atom]

        elif match := re.match(r'([cf])=(.+?),(.+?),(.+)', atom):
            coord = match.groups()[0]
            position = np.array(match.groups()[1:4], float)
            if coord.lower() == 'f':
                position = np.dot(position, lattice_vectors)
            for n, pos in enumerate(positions):
                if np.allclose(
                    position, pos, configuration.equal_cell_positions_tolerance
                ):
                    indices.append(n)
                    symbols.append(labels[n])

        elif isinstance(atom, str):
            indices = [n for n, label in enumerate(labels) if label == atom]
            symbols = [atom]

        return dict(label=''.join(symbols), indices=indices)

    def get_orbitals_state(self, orbital: Any) -> List[Dict[str, Any]]:
        if orbital is None:
            return None

        states = []
        orbitals = re.findall(r'l=([\d+])(?:,mr=([\d])+=)?', orbital)
        for orb in orbitals:
            nl = int(orb[0])
            states.append(dict(l=self._l_symbols[nl]))
            if orb[1]:
                nm = sum([len(range(-n, n + 1)) for n in range(nl)]) + int(orb[1])
                states[-1]['m'] = self._m_symbols[nm]
        if not orbitals:
            for orb in orbital.split(';'):
                try:
                    norb = self._wannier_symbols.index(orb)
                except Exception:
                    continue
                # calculate l,m from norb
                nl = 0
                nm = 0
                while True:
                    m_offset = [nm + nq for nq in range(len(range(-nl, nl + 1)))]
                    if norb in m_offset:
                        nm = m_offset.index(norb)
                        break
                    nl += 1
                    nm += len(m_offset)
                states.append(dict(l=self._l_symbols[nl], m=self._m_symbols[nm]))
        return states


class Wannier90Parser:
    def get_dft_files(self, mainfile: str) -> List[str]:
        for filename in ['vasprun.xml', 'OUTCAR']:
            files = get_files(
                pattern=f'*{filename}',
                filepath=mainfile,
                stripname=os.path.basename(mainfile),
                deep=False,
            )
            if files:
                return files
        return []

    def get_mainfile_keys(self, **kwargs):
        """
        Generates extra `child_archives` to create the DFT+TB workflow if the conditions in `workflow_dft_files` are met.
        """
        dft_files = self.get_dft_files(kwargs.get('filename', ''))
        return ['DFTPlusTB_workflow'] if dft_files else True

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: Dict[str, 'EntryArchive'] = {},
    ) -> None:
        basename = os.path.basename(mainfile)
        # define mapping parser interface to OutParser
        wout_parser = WOutTextParser(text_parser=WOutParser())
        wout_parser.filepath = mainfile

        # construct metainfo parser
        data = Simulation()
        data_parser = MetainfoParser()
        data_parser.annotation_key = OUT_ANNOTATION_KEY
        data_parser.data_object = data

        wout_parser.convert(data_parser)
        archive.data = data

        # parse input file
        win_parser = WInTextParser(text_parser=WInParser())
        if data.model_system:
            win_files = get_files(
                pattern='*.win', filepath=mainfile, stripname=basename
            )
            if len(win_files) > 1:
                logger.warning(
                    'Multiple `*.win` files found. We will parse the first one.'
                )
            if win_files:
                win_parser.filepath = win_files[0]
                # need data from out
                for key in ['structure', 'lattice_vectors']:
                    win_parser.data[key] = wout_parser.data.get(key)
                data_parser.annotation_key = IN_ANNOTATION_KEY
                data_parser.data_object = data
                win_parser.convert(data_parser)

        # parse hr files
        whr_parser = WHrTextParser(text_parser=HrParser())
        hr_files = get_files(pattern='*hr.dat', filepath=mainfile, stripname=basename)
        if len(hr_files) > 1:
            logger.info('Multiple `*hr.dat` files found.')
        for hr_file in hr_files:
            whr_parser.filepath = hr_file
            # need data from out
            whr_parser.data['n_orbitals'] = wout_parser.data.get('Nwannier')
            data_parser.annotation_key = HR_ANNOTATION_KEY
            data_parser.data_object = data
            whr_parser.convert(data_parser)

        # parse dos files
        wdos_parser = WDosTextParser(text_parser=DataTextParser())
        dos_files = get_files(pattern='*dos.dat', filepath=mainfile, stripname=basename)
        if len(dos_files) > 1:
            logger.info('Multiple `*dos.dat` files found.')
        for dos_file in dos_files:
            wdos_parser.filepath = dos_file
            wdos_parser.data_object.parse('data')
            data_parser.annotation_key = DOS_ANNOTATION_KEY
            data_parser.data_object = data
            wdos_parser.convert(data_parser)

        wband_parser = WBandTextParser(text_parser=DataTextParser())
        # parse band files
        band_files = get_files(
            pattern='*band.dat', filepath=mainfile, stripname=basename
        )
        for band_file in band_files:
            wband_parser.filepath = band_file
            wband_parser.data_object.parse('data')
            data_parser.annotation_key = BAND_ANNOTATION_KEY
            data_parser.data_object = data
            wband_parser.convert(data_parser)

        workflow = SinglePoint()
        workflow.normalize(archive=archive, logger=logger)
        archive.workflow2 = workflow

        # workflow
        if child_archives:
            from nomad.app.v1.routers.uploads import get_upload_with_read_access
            from nomad.datamodel import User

            upload_id = archive.metadata.upload_id
            upload = get_upload_with_read_access(
                upload_id=upload_id,
                user=User.get(user_id=archive.metadata.main_author.user_id),
            )
            dft_archive = None
            dft_files = self.get_dft_files(mainfile)
            dft_path = dft_files[-1].split('raw/')[-1]
            with upload.entries_metadata() as entries_metadata:
                for metadata in entries_metadata:
                    if metadata.mainfile == dft_path:
                        dft_archive = upload.get_entry(
                            metadata.entry_id
                        )._parser_results
                        break
            dft_plus_tb_archive = child_archives.get('DFTPlusTB_workflow')
            dft_plus_tb = parse_dft_plus_tb_workflow(
                dft_archive=dft_archive, tb_archive=archive
            )
            dft_plus_tb_archive.workflow2 = dft_plus_tb

        # debug
        # self.wout_parser = wout_parser
        # self.data_parser = data_parser
        # self.win_parser = win_parser
        # self.wdos_parser = wdos_parser
        # self.wband_parser = wband_parser
        # self.whr_parser = whr_parser
        # close parser contexts
        wout_parser.close()
        data_parser.close()
        win_parser.close()
        wdos_parser.close()
        wband_parser.close()
        whr_parser.close()
