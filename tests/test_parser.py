#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

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


def test_single_point_La2CuO4(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join(os.path.dirname(__file__), 'data/lco_mlwf/lco.wout'),
        archive,
        logger,
    )
    simulation = archive.data
    assert simulation.program.name == 'Wannier90'
    assert simulation.program.version == '3.1.0'


def test_single_point_LK99(parser):
    # archive = EntryArchive()
    assert True
    # parser.parse(
    #     os.path.join(os.path.dirname(__file__), 'data/lk99_liangsi_1/k000.wout'),
    #     archive,
    #     logger,
    # )
    # simulation = archive.data
    # assert simulation.program.name == 'Wannier90'
    # assert simulation.program.version == '2.0.1'
