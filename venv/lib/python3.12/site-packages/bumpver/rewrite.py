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
import difflib
str = getattr(builtins, 'unicode', str)
from bumpver import pathlib as pl
from . import config
from .patterns import Pattern


class NoPatternMatch(Exception):
    """Pattern not found in content.

    logger.error is used to show error info about the patterns so
    that users can debug what is wrong with them. The class
    itself doesn't capture that info. This approach is used so
    that all patter issues can be shown, rather than bubbling
    all the way up the stack on the very first pattern with no
    matches.
    """


def detect_line_sep(content):
    """Parse line separator from content.

    >>> detect_line_sep('\\r\\n')
    '\\r\\n'
    >>> detect_line_sep('\\r')
    '\\r'
    >>> detect_line_sep('\\n')
    '\\n'
    >>> detect_line_sep('')
    '\\n'
    """
    if '\r\n' in content:
        return '\r\n'
    elif '\r' in content:
        return '\r'
    else:
        return '\n'


RewrittenFileData = typ.NamedTuple('RewrittenFileData', [('path', str), (
    'line_sep', str), ('old_lines', typ.List[str]), ('new_lines', typ.List[
    str])])
PathPatternsItem = typ.Tuple[pl.Path, typ.List[Pattern]]


def iter_path_patterns_items(file_patterns):
    for filepath_str, patterns in file_patterns.items():
        filepath_obj = pl.Path(filepath_str)
        if filepath_obj.exists():
            yield filepath_obj, patterns
        else:
            errmsg = "File does not exist: '{0}'".format(filepath_str)
            raise IOError(errmsg)


def diff_lines(rfd):
    """Generate unified diff.

    >>> rfd = RewrittenFileData(
    ...    path      = "<path>",
    ...    line_sep  = "\\n",
    ...    old_lines = ["foo"],
    ...    new_lines = ["bar"],
    ... )
    >>> diff_lines(rfd)
    ['--- <path>', '+++ <path>', '@@ -1 +1 @@', '-foo', '+bar']
    """
    lines = difflib.unified_diff(a=rfd.old_lines, b=rfd.new_lines, lineterm
        ='', fromfile=rfd.path, tofile=rfd.path)
    return list(lines)
