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
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import typing as typ
str = getattr(builtins, 'unicode', str)
Pattern = typ.NamedTuple('Pattern', [('version_pattern', str), (
    'raw_pattern', str), ('regexp', typ.Pattern[str])])
RE_PATTERN_ESCAPES = [('\\', '\\\\'), ('-', '\\-'), ('.', '\\.'), ('+',
    '\\+'), ('*', '\\*'), ('?', '\\?'), ('{', '\\{'), ('}', '\\}'), ('[',
    '\\['), (']', '\\]'), ('(', '\\('), (')', '\\)')]
