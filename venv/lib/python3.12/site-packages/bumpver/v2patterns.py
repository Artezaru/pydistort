# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""Compose Regular Expressions from Patterns.

>>> pattern = compile_pattern("vYYYY0M.BUILD[-TAG]")
>>> version_info = pattern.regexp.match("v201712.0123-alpha")
>>> assert version_info.groupdict() == {
...     "year_y" : "2017",
...     "month"  : "12",
...     "bid"    : "0123",
...     "tag"    : "alpha",
... }
>>>
>>> version_info = pattern.regexp.match("201712.1234")
>>> assert version_info is None

>>> version_info = pattern.regexp.match("v201713.1234")
>>> assert version_info is None

>>> version_info = pattern.regexp.match("v201712.1234")
>>> assert version_info.groupdict() == {
...     "year_y" : "2017",
...     "month"  : "12",
...     "bid"    : "1234",
...     "tag"    : None,
... }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import typing as typ
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import collections
str = getattr(builtins, 'unicode', str)
from . import utils
from .patterns import RE_PATTERN_ESCAPES
from .patterns import Pattern
PART_PATTERNS = collections.OrderedDict([('YYYY', '[1-9][0-9]{3}'), ('YY',
    '[1-9][0-9]?'), ('0Y', '[0-9]{2}'), ('GGGG', '[1-9][0-9]{3}'), ('GG',
    '[1-9][0-9]?'), ('0G', '[0-9]{2}'), ('Q', '[1-4]'), ('MM',
    '1[0-2]|[1-9]'), ('0M', '1[0-2]|0[1-9]'), ('DD',
    '3[0-1]|[1-2][0-9]|[1-9]'), ('0D', '3[0-1]|[1-2][0-9]|0[1-9]'), ('JJJ',
    '36[0-6]|3[0-5][0-9]|[1-2][0-9][0-9]|[1-9][0-9]|[1-9]'), ('00J',
    '36[0-6]|3[0-5][0-9]|[1-2][0-9][0-9]|0[1-9][0-9]|00[1-9]'), ('WW',
    '5[0-2]|[1-4][0-9]|[0-9]'), ('0W', '5[0-2]|[0-4][0-9]'), ('UU',
    '5[0-2]|[1-4][0-9]|[0-9]'), ('0U', '5[0-2]|[0-4][0-9]'), ('VV',
    '5[0-3]|[1-4][0-9]|[1-9]'), ('0V', '5[0-3]|[1-4][0-9]|0[1-9]'), (
    'MAJOR', '[0-9]+'), ('MINOR', '[0-9]+'), ('PATCH', '[0-9]+'), ('BUILD',
    '[0-9]+'), ('BLD', '[1-9][0-9]*'), ('TAG',
    'preview|final|dev|alpha|beta|post|rc'), ('PYTAG', 'dev|post|rc|a|b'),
    ('GITHASH', '\\.[0-9]+\\+.*'), ('HEXHASH', '[0-9a-f]+'), ('NUM',
    '[0-9]+'), ('INC0', '[0-9]+'), ('INC1', '[1-9][0-9]*')])
PATTERN_PART_FIELDS = {'YYYY': 'year_y', 'YY': 'year_y', '0Y': 'year_y',
    'GGGG': 'year_g', 'GG': 'year_g', '0G': 'year_g', 'Q': 'quarter', 'MM':
    'month', '0M': 'month', 'DD': 'dom', '0D': 'dom', 'JJJ': 'doy', '00J':
    'doy', 'MAJOR': 'major', 'MINOR': 'minor', 'PATCH': 'patch', 'BUILD':
    'bid', 'BLD': 'bid', 'TAG': 'tag', 'PYTAG': 'pytag', 'GITHASH':
    'githash', 'HEXHASH': 'hexhash', 'NUM': 'num', 'INC0': 'inc0', 'INC1':
    'inc1', 'WW': 'week_w', '0W': 'week_w', 'UU': 'week_u', '0U': 'week_u',
    'VV': 'week_v', '0V': 'week_v'}
PEP440_PART_SUBSTITUTIONS = {'0W': 'WW', '0U': 'UU', '0V': 'VV', '0M': 'MM',
    '0D': 'DD', '00J': 'JJJ', 'BUILD': 'BLD', 'TAG': 'PYTAG'}
FieldValue = typ.Union[str, int]


def _fmt_num(val):
    return str(val)


def _fmt_bld(val):
    return str(int(val))


def _fmt_yy(year_y):
    return str(int(str(year_y)[-2:]))


def _fmt_0y(year_y):
    return '{0:02}'.format(int(str(year_y)[-2:]))


def _fmt_gg(year_g):
    return str(int(str(year_g)[-2:]))


def _fmt_0g(year_g):
    return '{0:02}'.format(int(str(year_g)[-2:]))


def _fmt_0m(month):
    return '{0:02}'.format(int(month))


def _fmt_0d(dom):
    return '{0:02}'.format(int(dom))


def _fmt_00j(doy):
    return '{0:03}'.format(int(doy))


def _fmt_0w(week_w):
    return '{0:02}'.format(int(week_w))


def _fmt_0u(week_u):
    return '{0:02}'.format(int(week_u))


def _fmt_0v(week_v):
    return '{0:02}'.format(int(week_v))


FormatterFunc = typ.Callable[[FieldValue], str]
PART_FORMATS = {'YYYY': _fmt_num, 'YY': _fmt_yy, '0Y': _fmt_0y, 'GGGG':
    _fmt_num, 'GG': _fmt_gg, '0G': _fmt_0g, 'Q': _fmt_num, 'MM': _fmt_num,
    '0M': _fmt_0m, 'DD': _fmt_num, '0D': _fmt_0d, 'JJJ': _fmt_num, '00J':
    _fmt_00j, 'MAJOR': _fmt_num, 'MINOR': _fmt_num, 'PATCH': _fmt_num,
    'BUILD': _fmt_num, 'BLD': _fmt_bld, 'TAG': _fmt_num, 'PYTAG': _fmt_num,
    'GITHASH': _fmt_num, 'HEXHASH': _fmt_num, 'NUM': _fmt_num, 'INC0':
    _fmt_num, 'INC1': _fmt_num, 'WW': _fmt_num, '0W': _fmt_0w, 'UU':
    _fmt_num, '0U': _fmt_0u, 'VV': _fmt_num, '0V': _fmt_0v}


def _convert_to_pep440(version_pattern):
    pep440_pattern = version_pattern
    if pep440_pattern.startswith('v'):
        pep440_pattern = pep440_pattern[1:]
    pep440_pattern = pep440_pattern.replace('\\[', '')
    pep440_pattern = pep440_pattern.replace('\\]', '')
    pep440_pattern, _ = re.subn('[^a-zA-Z0-9\\.\\!\\[\\]]', '', pep440_pattern)
    part_names = list(PATTERN_PART_FIELDS.keys())
    part_names.sort(key=len, reverse=True)
    for part_name in part_names:
        if part_name not in version_pattern:
            continue
        if part_name not in PEP440_PART_SUBSTITUTIONS:
            continue
        substitution = PEP440_PART_SUBSTITUTIONS[part_name]
        if substitution in pep440_pattern:
            continue
        is_numerical_part = part_name not in ('TAG', 'PYTAG')
        if is_numerical_part:
            part_index = pep440_pattern.find(part_name)
            is_zero_truncation_part = part_index == 0 or pep440_pattern[
                part_index - 1] == '.'
            if is_zero_truncation_part:
                pep440_pattern = pep440_pattern.replace(part_name, substitution
                    )
        else:
            pep440_pattern = pep440_pattern.replace(part_name, substitution)
    if 'PYTAGNUM' not in pep440_pattern:
        pep440_pattern = pep440_pattern.replace('PYTAG', '')
        pep440_pattern = pep440_pattern.replace('NUM', '')
        pep440_pattern = pep440_pattern.replace('[]', '')
        pep440_pattern += '[PYTAGNUM]'
    return pep440_pattern


def normalize_pattern(version_pattern, raw_pattern):
    normalized_pattern = raw_pattern
    if '{version}' in raw_pattern:
        normalized_pattern = normalized_pattern.replace('{version}',
            version_pattern)
    if '{pep440_version}' in normalized_pattern:
        pep440_version_pattern = _convert_to_pep440(version_pattern)
        normalized_pattern = normalized_pattern.replace('{pep440_version}',
            pep440_version_pattern)
    return normalized_pattern


SortKey = typ.Tuple[int, int]
PostitionedPart = typ.Tuple[int, int, str]


def _iter_part_patterns(pattern):
    used_fields = set()
    for part_name, part_pattern in PART_PATTERNS.items():
        end_idx = 0
        while True:
            start_idx = pattern.find(part_name, end_idx)
            if start_idx < 0:
                break
            field = PATTERN_PART_FIELDS[part_name]
            if field in used_fields:
                named_part_pattern = '(?P<{0}_{1}>{2})'.format(field, len(
                    used_fields), part_pattern)
            else:
                named_part_pattern = '(?P<{0}>{1})'.format(field, part_pattern)
            used_fields.add(field)
            end_idx = start_idx + len(part_name)
            sort_key = -end_idx, -len(part_name)
            positioned_part = start_idx, end_idx, named_part_pattern
            yield sort_key, positioned_part


def _replace_pattern_parts(pattern):
    while True:
        new_pattern, _n = re.subn('([^\\\\]|^)\\[', '\\1(?:', pattern)
        new_pattern, _m = re.subn('([^\\\\]|^)\\]', '\\1)?', new_pattern)
        pattern = new_pattern
        if _n + _m == 0:
            break
    part_patterns_by_index = dict(_iter_part_patterns(pattern))
    last_start_idx = len(pattern) + 1
    result_pattern = pattern
    for _, (start_idx, end_idx, named_part_pattern) in sorted(
        part_patterns_by_index.items()):
        if end_idx <= last_start_idx:
            result_pattern = result_pattern[:start_idx
                ] + named_part_pattern + result_pattern[end_idx:]
            last_start_idx = start_idx
    return result_pattern


def _compile_pattern_re(normalized_pattern):
    escaped_pattern = normalized_pattern
    for char, escaped in RE_PATTERN_ESCAPES:
        is_semantic_char = char in '[]\\'
        if not is_semantic_char:
            escaped_pattern = escaped_pattern.replace(char, escaped)
    pattern_str = _replace_pattern_parts(escaped_pattern)
    return re.compile(pattern_str)


@utils.memo
def compile_pattern(version_pattern, raw_pattern=None):
    _raw_pattern = version_pattern if raw_pattern is None else raw_pattern
    normalized_pattern = normalize_pattern(version_pattern, _raw_pattern)
    regexp = _compile_pattern_re(normalized_pattern)
    return Pattern(version_pattern, normalized_pattern, regexp)


def compile_patterns(version_pattern, raw_patterns):
    return [compile_pattern(version_pattern, raw_pattern) for raw_pattern in
        raw_patterns]
