# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""Parse PyCalVer strings from files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import typing as typ
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
from .patterns import Pattern
str = getattr(builtins, 'unicode', str)
LineNo = int
Start = int
End = int
LineSpan = typ.NamedTuple('LineSpan', [('lineno', LineNo), ('start', Start),
    ('end', End)])
LineSpans = typ.List[LineSpan]


def _has_overlap(needle, haystack):
    for span in haystack:
        has_overlap = (span.lineno == needle.lineno and needle.start <=
            span.end and needle.end >= span.start)
        if has_overlap:
            return True
    return False


PatternMatch = typ.NamedTuple('PatternMatch', [('lineno', LineNo), ('line',
    str), ('pattern', Pattern), ('span', typ.Tuple[Start, End]), ('match',
    str)])
PatternMatches = typ.Iterable[PatternMatch]


def _iter_for_pattern(lines, pattern):
    for lineno, line in enumerate(lines):
        match = pattern.regexp.search(line)
        if match and len(match.group(0)) > 0:
            yield PatternMatch(lineno, line, pattern, match.span(), match.
                group(0))


def iter_matches(lines, patterns):
    """Iterate over all matches of any pattern on any line.

    >>> from . import v1patterns
    >>> lines = ["__version__ = 'v201712.0002-alpha'"]
    >>> version_pattern = "{pycalver}"
    >>> raw_patterns = ["{pycalver}", "{pep440_pycalver}"]
    >>> patterns = [v1patterns.compile_pattern(version_pattern, p) for p in raw_patterns]
    >>> matches = list(iter_matches(lines, patterns))
    >>> assert matches[0] == PatternMatch(
    ...     lineno = 0,
    ...     line   = "__version__ = 'v201712.0002-alpha'",
    ...     pattern= v1patterns.compile_pattern(version_pattern),
    ...     span   = (15, 33),
    ...     match  = "v201712.0002-alpha",
    ... )
    """
    matched_spans = []
    for pattern in patterns:
        for match in _iter_for_pattern(lines, pattern):
            needle_span = LineSpan(match.lineno, *match.span)
            if not _has_overlap(needle_span, matched_spans):
                yield match
            matched_spans.append(needle_span)
