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
import sys
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import typing as typ
str = getattr(builtins, 'unicode', str)
PY2 = sys.version_info.major < 3
try:
    try:
        from urllib.parse import quote as py3_stdlib_quote
    except ImportError:
        from urlparse import quote as py3_stdlib_quote
except ImportError:
    from urllib import quote as py2_stdlib_quote


def quote(string, safe='/', encoding=None, errors=None):
    if not isinstance(string, str):
        errmsg = 'Expected str/unicode but got {0}'.format(type(string))
        raise TypeError(errmsg)
    if encoding is None:
        _encoding = 'utf-8'
    else:
        _encoding = encoding
    if errors is None:
        _errors = 'strict'
    else:
        _errors = errors
    if PY2:
        data = string.encode(_encoding)
        res = py2_stdlib_quote(data, safe=safe.encode(_encoding))
        return res.decode(_encoding, errors=_errors)
    else:
        return py3_stdlib_quote(string, safe=safe, encoding=_encoding,
            errors=_errors)
