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
import typing as typ
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import datetime as dt
str = getattr(builtins, 'unicode', str)
from . import utils
from . import setuptools_v65_version
MaybeInt = typ.Optional[int]


def parse_version(version):
    return setuptools_v65_version.parse(version)


V1CalendarInfo = typ.NamedTuple('V1CalendarInfo', [('year', MaybeInt), (
    'quarter', MaybeInt), ('month', MaybeInt), ('dom', MaybeInt), ('doy',
    MaybeInt), ('iso_week', MaybeInt), ('us_week', MaybeInt)])
V1VersionInfo = typ.NamedTuple('V1VersionInfo', [('year', MaybeInt), (
    'quarter', MaybeInt), ('month', MaybeInt), ('dom', MaybeInt), ('doy',
    MaybeInt), ('iso_week', MaybeInt), ('us_week', MaybeInt), ('major', int
    ), ('minor', int), ('patch', int), ('bid', str), ('tag', str)])
V2CalendarInfo = typ.NamedTuple('V2CalendarInfo', [('year_y', MaybeInt), (
    'year_g', MaybeInt), ('quarter', MaybeInt), ('month', MaybeInt), ('dom',
    MaybeInt), ('doy', MaybeInt), ('week_w', MaybeInt), ('week_u', MaybeInt
    ), ('week_v', MaybeInt)])
V2VersionInfo = typ.NamedTuple('V2VersionInfo', [('year_y', MaybeInt), (
    'year_g', MaybeInt), ('quarter', MaybeInt), ('month', MaybeInt), ('dom',
    MaybeInt), ('doy', MaybeInt), ('week_w', MaybeInt), ('week_u', MaybeInt
    ), ('week_v', MaybeInt), ('major', int), ('minor', int), ('patch', int),
    ('bid', str), ('tag', str), ('pytag', str), ('githash', str), (
    'hexhash', str), ('num', int), ('inc0', int), ('inc1', int)])
TODAY = utils.now().date()
TAG_BY_PEP440_TAG = {'a': 'alpha', 'b': 'beta', '': 'final', 'rc': 'rc',
    'dev': 'dev', 'post': 'post'}
PEP440_TAG_BY_TAG = {'a': 'a', 'b': 'b', 'dev': 'dev', 'alpha': 'a', 'beta':
    'b', 'preview': 'rc', 'pre': 'rc', 'rc': 'rc', 'c': 'rc', 'final': '',
    'post': 'post', 'r': 'post', 'rev': 'post'}
assert set(TAG_BY_PEP440_TAG.keys()) == set(PEP440_TAG_BY_TAG.values())
assert set(TAG_BY_PEP440_TAG.values()) < set(PEP440_TAG_BY_TAG.keys())
PART_ZERO_VALUES = {'MAJOR': '0', 'MINOR': '0', 'PATCH': '0', 'TAG':
    'final', 'PYTAG': '', 'NUM': '0', 'INC0': '0'}
V2_FIELD_INITIAL_VALUES = {'major': '0', 'minor': '0', 'patch': '0', 'num':
    '0', 'inc0': '0', 'inc1': '1'}


def is_zero_val(part, part_value):
    return part in PART_ZERO_VALUES and part_value == PART_ZERO_VALUES[part]


class PatternError(Exception):
    pass


def date_from_doy(year, doy):
    """Parse date from year and day of year (1 indexed).

    >>> cases = [
    ...     (2016, 1), (2016, 31), (2016, 31 + 1), (2016, 31 + 29), (2016, 31 + 30),
    ...     (2017, 1), (2017, 31), (2017, 31 + 1), (2017, 31 + 28), (2017, 31 + 29),
    ... ]
    >>> dates = [date_from_doy(year, month) for year, month in cases]
    >>> assert [(d.month, d.day) for d in dates] == [
    ...     (1, 1), (1, 31), (2, 1), (2, 29), (3, 1),
    ...     (1, 1), (1, 31), (2, 1), (2, 28), (3, 1),
    ... ]
    """
    return dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)


def quarter_from_month(month):
    """Calculate quarter (1 indexed) from month (1 indexed).

    >>> [quarter_from_month(month) for month in range(1, 13)]
    [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    """
    return (month - 1) // 3 + 1


def to_pep440(version):
    """Derive pep440 compliant version string from PyCalVer version string.

    >>> to_pep440("v201811.0007-beta")
    '201811.7b0'
    """
    return str(parse_version(version))
