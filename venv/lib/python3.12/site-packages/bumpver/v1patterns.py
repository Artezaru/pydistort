# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""Compose Regular Expressions from Patterns.

>>> version_info = PYCALVER_RE.match("v201712.0123-alpha").groupdict()
>>> assert version_info == {
...     "pycalver"    : "v201712.0123-alpha",
...     "vYYYYMM"     : "v201712",
...     "year"        : "2017",
...     "month"       : "12",
...     "build"       : ".0123",
...     "build_no"    : "0123",
...     "release"     : "-alpha",
...     "release_tag" : "alpha",
... }
>>>
>>> version_info = PYCALVER_RE.match("v201712.0033").groupdict()
>>> assert version_info == {
...     "pycalver"   : "v201712.0033",
...     "vYYYYMM"    : "v201712",
...     "year"       : "2017",
...     "month"      : "12",
...     "build"      : ".0033",
...     "build_no"   : "0033",
...     "release"    : None,
...     "release_tag": None,
... }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import typing as typ
import logging
from . import utils
from .patterns import RE_PATTERN_ESCAPES
from .patterns import Pattern
logger = logging.getLogger('bumpver.v1patterns')
PYCALVER_PATTERN = """
\\b
(?P<pycalver>
    (?P<vYYYYMM>
       v                        # "v" version prefix
       (?P<year>\\d{4})
       (?P<month>\\d{2})
    )
    (?P<build>
        \\.                      # "." build nr prefix
        (?P<build_no>\\d{4,})
    )
    (?P<release>
        \\-                      # "-" release prefix
        (?P<release_tag>alpha|beta|dev|rc|post)
    )?
)(?:\\s|$)
"""
PYCALVER_RE = re.compile(PYCALVER_PATTERN, flags=re.VERBOSE)
COMPOSITE_PART_PATTERNS = {'pep440_pycalver':
    '{year}{month}\\.{BID}(?:{pep440_tag})?', 'pycalver':
    'v{year}{month}\\.{bid}(?:-{tag})?', 'calver': 'v{year}{month}',
    'semver': '{MAJOR}\\.{MINOR}\\.{PATCH}', 'release_tag': '{tag}',
    'build': '\\.{bid}', 'release': '(?:-{tag})?', 'pep440_version':
    '{year}{month}\\.{BID}(?:{pep440_tag})?'}
PART_PATTERNS = {'year': '\\d{4}', 'month': '(?:0[0-9]|1[0-2])',
    'month_short': '(?:1[0-2]|[1-9])', 'build_no': '\\d{4,}', 'pep440_tag':
    '(?:a|b|dev|rc|post)?\\d*', 'tag': '(?:alpha|beta|dev|rc|post|final)',
    'yy': '\\d{2}', 'yyyy': '\\d{4}', 'quarter': '[1-4]', 'iso_week':
    '(?:[0-4]\\d|5[0-3])', 'us_week': '(?:[0-4]\\d|5[0-3])', 'dom':
    '(0[1-9]|[1-2][0-9]|3[0-1])', 'dom_short': '([1-9]|[1-2][0-9]|3[0-1])',
    'doy': '(?:[0-2]\\d\\d|3[0-5][0-9]|36[0-6])', 'doy_short':
    '(?:[0-2]\\d\\d|3[0-5][0-9]|36[0-6])', 'MAJOR': '\\d+', 'MINOR': '\\d+',
    'MM': '\\d{2,}', 'MMM': '\\d{3,}', 'MMMM': '\\d{4,}', 'MMMMM':
    '\\d{5,}', 'PATCH': '\\d+', 'PP': '\\d{2,}', 'PPP': '\\d{3,}', 'PPPP':
    '\\d{4,}', 'PPPPP': '\\d{5,}', 'bid': '\\d{4,}', 'BID': '[1-9]\\d*',
    'BB': '[1-9]\\d{1,}', 'BBB': '[1-9]\\d{2,}', 'BBBB': '[1-9]\\d{3,}',
    'BBBBB': '[1-9]\\d{4,}', 'BBBBBB': '[1-9]\\d{5,}', 'BBBBBBB':
    '[1-9]\\d{6,}'}
PATTERN_PART_FIELDS = {'year': 'year', 'month': 'month', 'month_short':
    'month', 'pep440_tag': 'tag', 'tag': 'tag', 'yy': 'year', 'yyyy':
    'year', 'quarter': 'quarter', 'iso_week': 'iso_week', 'us_week':
    'us_week', 'dom': 'dom', 'doy': 'doy', 'dom_short': 'dom', 'doy_short':
    'doy', 'MAJOR': 'major', 'MINOR': 'minor', 'MM': 'minor', 'MMM':
    'minor', 'MMMM': 'minor', 'MMMMM': 'minor', 'PP': 'patch', 'PPP':
    'patch', 'PPPP': 'patch', 'PPPPP': 'patch', 'PATCH': 'patch',
    'build_no': 'bid', 'bid': 'bid', 'BID': 'bid', 'BB': 'bid', 'BBB':
    'bid', 'BBBB': 'bid', 'BBBBB': 'bid', 'BBBBBB': 'bid', 'BBBBBBB': 'bid'}
FULL_PART_FORMATS = {'pep440_pycalver':
    '{year}{month:02}.{BID}{pep440_tag}', 'pycalver':
    'v{year}{month:02}.{bid}{release}', 'calver': 'v{year}{month:02}',
    'semver': '{MAJOR}.{MINOR}.{PATCH}', 'release_tag': '{tag}', 'build':
    '.{bid}', 'month': '{month:02}', 'month_short': '{month}', 'build_no':
    '{bid}', 'iso_week': '{iso_week:02}', 'us_week': '{us_week:02}', 'dom':
    '{dom:02}', 'doy': '{doy:03}', 'dom_short': '{dom}', 'doy_short':
    '{doy}', 'pep440_version': '{year}{month:02}.{BID}{pep440_tag}',
    'version': 'v{year}{month:02}.{bid}{release}'}


def _replace_pattern_parts(pattern):
    for part_name, part_pattern in PART_PATTERNS.items():
        named_part_pattern = '(?P<{0}>{1})'.format(part_name, part_pattern)
        placeholder = '\\{' + part_name + '\\}'
        pattern = pattern.replace(placeholder, named_part_pattern)
    return pattern


def _init_composite_patterns():
    for part_name, part_pattern in COMPOSITE_PART_PATTERNS.items():
        part_pattern = part_pattern.replace('{', '\\{').replace('}', '\\}')
        pattern_str = _replace_pattern_parts(part_pattern)
        PART_PATTERNS[part_name] = pattern_str


_init_composite_patterns()


def _compile_pattern_re(normalized_pattern):
    escaped_pattern = normalized_pattern
    for char, escaped in RE_PATTERN_ESCAPES:
        escaped_pattern = escaped_pattern.replace(char, escaped)
    pattern_str = _replace_pattern_parts(escaped_pattern)
    return re.compile(pattern_str)


def _normalized_pattern(version_pattern, raw_pattern):
    res = raw_pattern.replace('{version}', version_pattern)
    if version_pattern == '{pycalver}':
        res = res.replace('{pep440_version}', '{pep440_pycalver}')
    elif version_pattern == '{semver}':
        res = res.replace('{pep440_version}', '{semver}')
    elif version_pattern == 'v{year}{month}{build}{release}':
        res = res.replace('{pep440_version}', '{year}{month}.{BID}{pep440_tag}'
            )
    elif version_pattern == '{year}{month}{build}{release}':
        res = res.replace('{pep440_version}', '{year}{month}.{BID}{pep440_tag}'
            )
    elif version_pattern == 'v{year}{build}{release}':
        res = res.replace('{pep440_version}', '{year}.{BID}{pep440_tag}')
    elif version_pattern == '{year}{build}{release}':
        res = res.replace('{pep440_version}', '{year}.{BID}{pep440_tag}')
    elif '{pep440_version}' in raw_pattern:
        logger.warning("No mapping of '{0}' to '{pep440_version}'".format(
            version_pattern))
    return res


@utils.memo
def compile_pattern(version_pattern, raw_pattern=None):
    _raw_pattern = version_pattern if raw_pattern is None else raw_pattern
    normalized_pattern = _normalized_pattern(version_pattern, _raw_pattern)
    regexp = _compile_pattern_re(normalized_pattern)
    return Pattern(version_pattern, normalized_pattern, regexp)


def compile_patterns(version_pattern, raw_patterns):
    return [compile_pattern(version_pattern, raw_pattern) for raw_pattern in
        raw_patterns]
