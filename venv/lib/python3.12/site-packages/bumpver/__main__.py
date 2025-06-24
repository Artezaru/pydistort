#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""
__main__ module for BumpVer.

Enables use as module: $ python -m bumpver --version
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from . import cli
if __name__ == '__main__':
    cli.cli()
