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
import datetime as dt
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import functools
str = getattr(builtins, 'unicode', str)


def memo(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        key = str(args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]
    return wrapper


def now():
    if hasattr(dt, 'UTC'):
        return dt.datetime.now(dt.UTC)
    else:
        return dt.datetime.utcnow()
