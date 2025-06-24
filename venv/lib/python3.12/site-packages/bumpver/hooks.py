# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import logging
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import subprocess as sp
str = getattr(builtins, 'unicode', str)
from bumpver import pathlib as pl
logger = logging.getLogger('bumpver.hooks')


def run(path, old_version, new_version):
    env = dict(os.environ, BUMPVER_OLD_VERSION=old_version,
        BUMPVER_NEW_VERSION=new_version)
    try:
        proc = sp.Popen(str(pl.Path(path).absolute()), env=env, stdout=sp.
            PIPE, stderr=sp.PIPE)
        if proc.stdout is not None:
            with proc.stdout as out:
                for line in iter(out.readline, b''):
                    logger.info('\t{0}'.format(line.decode('utf8').strip()))
        if proc.stderr is not None:
            with proc.stderr as err:
                for line in iter(err.readline, b''):
                    logger.error('\t{0}'.format(line.decode('utf8').strip()))
        proc.wait()
    except IOError as err:
        logger.error('\t{0}'.format(err))
        logger.error('Script exited with an error. Stopping')
        sys.exit(1)
    if proc.returncode != 0:
        logger.error('Script exited with an error. Stopping')
        sys.exit(1)
