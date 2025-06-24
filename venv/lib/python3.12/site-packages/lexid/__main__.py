#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the lexid project
# https://github.com/mbarkhau/lexid
#
# Copyright (c) 2020-2021 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import click
from lexid import next_id
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
from lexid import ord_val
range = getattr(builtins, 'xrange', range)
try:
    import pretty_traceback
    pretty_traceback.install()
except ImportError:
    pass
click.disable_unicode_literals_warning = True


@click.command()
@click.option('-n', '--num', default=1)
@click.option('--debug', is_flag=True, default=False)
@click.argument('start_id', default='1001')
def main(start_id='1001', num=1, debug=False):
    """Increment a lexid."""
    if debug:
        print('{0:<13} {1:>12}'.format('lexical', 'numerical'))
    _curr_id = start_id
    for _ in range(num):
        try:
            _next_id = next_id(_curr_id)
        except OverflowError as err:
            sys.stderr.write('OverflowError: {0}'.format(err))
            sys.exit(1)
        if debug:
            print('{0:<13} {1:>12}'.format(_next_id, ord_val(_next_id)))
        else:
            print(_next_id)
        _curr_id = _next_id
