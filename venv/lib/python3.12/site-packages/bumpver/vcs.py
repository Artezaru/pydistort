# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
#
# bumpver/vcs.py (this file) is based on code from the
# bumpversion project: https://github.com/peritus/bumpversion
# Copyright (c) 2013-2014 Filip Noetzel - MIT License

"""Minimal Git and Mercirial API.

If terminology for similar concepts differs between git and
mercurial, then the git terms are used. For example "fetch"
(git) instead of "pull" (hg) .
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import sys
import shlex
import typing as typ
import logging
import tempfile
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import subprocess as sp
str = getattr(builtins, 'unicode', str)
from . import hooks
from . import config
logger = logging.getLogger('bumpver.vcs')
BRANCH_PATTERN = """
    (?P<is_current>\\*)?
    \\s+
    (?P<branch>[\\S]+)
    \\s+
    [0-9a-f]+
    \\s+
    \\[(?P<remote>[^/]+)/[^\\]]+\\]
"""
BRANCH_RE = re.compile(BRANCH_PATTERN, flags=re.VERBOSE)
VCS_SUBCOMMANDS_BY_NAME = {'git': {'is_usable': 'git rev-parse --git-dir',
    'fetch': 'git fetch', 'ls_tags': 'git tag --list', 'ls_tags_branch':
    'git tag --list --merged', 'status': 'git status --porcelain',
    'add_path': "git add --update '{path}'", 'commit':
    "git commit --message '{message}'", 'tag':
    "git tag --annotate {tag} --message '{message}'", 'tag_light':
    'git tag {tag}', 'push_tag':
    'git push {remote} --follow-tags {tag} HEAD', 'push':
    'git push {remote} HEAD', 'show_remotes':
    'git config --get remote.origin.url', 'ls_branches': 'git branch -vv'},
    'hg': {'is_usable': 'hg root', 'fetch': 'hg pull', 'ls_tags': 'hg tags',
    'ls_tags_branch':
    "hg log --branch . --rev='tag()' --template='{{tags}}\\n'", 'status':
    'hg status -umard', 'add_path': "hg add '{path}'", 'commit':
    "hg commit --logfile '{path}'", 'tag':
    "hg tag {tag} --message '{message}'", 'tag_light': 'hg tag {tag}',
    'push_tag': 'hg push {tag}', 'push': 'hg push', 'show_remotes': 'hg paths'}
    }
Env = typ.Dict[str, str]


class VCSAPI(object):
    """Absraction for git and mercurial."""

    def __init__(self, name, subcommands=None):
        self.name = name
        if subcommands is None:
            self.subcommands = VCS_SUBCOMMANDS_BY_NAME[name]
        else:
            self.subcommands = subcommands

    def __call__(self, cmd_name, env=None, **kwargs):
        """Invoke subcommand and return output."""
        cmd_tmpl = self.subcommands[cmd_name]
        cmd_str = cmd_tmpl.format(**kwargs)
        if cmd_name in ('commit', 'tag', 'push_tag'):
            logger.info(cmd_str)
        else:
            logger.debug(cmd_str)
        cmd_parts = shlex.split(cmd_str)
        output_data = sp.check_output(cmd_parts, env=env, stderr=sp.PIPE)
        return output_data.decode('utf-8')

    @property
    def is_usable(self):
        """Detect availability of subcommand."""
        if not os.path.exists('.{0}'.format(self.name)):
            return False
        cmd = self.subcommands['is_usable'].split()
        try:
            retcode = sp.call(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
            return retcode == 0
        except OSError as err:
            if err.errno == 2:
                return False
            else:
                raise

    def get_remote(self):
        try:
            if self.name == 'git':
                output = self('ls_branches')
                for match in BRANCH_RE.finditer(output):
                    branch_info = match.groupdict()
                    if branch_info['is_current']:
                        return branch_info['remote']
            output = self('show_remotes')
            if output.strip() == '':
                return None
            else:
                return output.strip()
        except Exception:
            return None

    def fetch(self):
        """Fetch updates from remote origin."""
        if self.get_remote():
            self('fetch')

    def status(self, required_files):
        """Get status lines."""
        status_output = self('status')
        status_items = [line.split(' ', 1) for line in status_output.
            splitlines()]
        return [filepath.strip() for status, filepath in status_items if 
            filepath.strip() in required_files or status != '??']

    def ls_tags(self):
        """List vcs tags on all branches."""
        ls_tag_lines = self('ls_tags').splitlines()
        logger.debug('ls_tags output {0}'.format(ls_tag_lines))
        return [line.strip().split(' ', 1)[0] for line in ls_tag_lines]

    def ls_tags_branch(self):
        """List vcs tags on all branches."""
        ls_tag_lines = self('ls_tags_branch').splitlines()
        logger.debug('ls_tags_branch output {0}'.format(ls_tag_lines))
        return [line.strip().split(' ', 1)[0] for line in ls_tag_lines]

    def add(self, path):
        """Add updates to be included in next commit."""
        try:
            self('add_path', path=path)
        except sp.CalledProcessError as ex:
            if 'already tracked!' in str(ex):
                return
            else:
                raise

    def commit(self, message):
        """Commit added files."""
        env = os.environ.copy()
        if self.name == 'git':
            self('commit', env=env, message=message)
        else:
            message_data = message.encode('utf-8')
            tmp_file = tempfile.NamedTemporaryFile('wb', delete=False)
            try:
                assert ' ' not in tmp_file.name
                fobj = None
                with tmp_file as fobj:
                    fobj.write(message_data)
                env['HGENCODING'] = 'utf-8'
                self('commit', env=env, path=tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

    def tag(self, tag_name, tag_message):
        """Create a tag."""
        if tag_message:
            self('tag', tag=tag_name, message=tag_message)
        else:
            self('tag_light', tag=tag_name)

    def push_tag(self, tag_name):
        """Push changes to origin."""
        remote = self.get_remote()
        if remote:
            self('push_tag', tag=tag_name, remote=remote)

    def push(self):
        """Push changes to origin."""
        remote = self.get_remote()
        if remote:
            self('push', remote=remote)

    def __repr__(self):
        """Generate string representation."""
        return "VCSAPI(name='{0}')".format(self.name)


def get_vcs_api():
    """Detect the appropriate VCS for a repository.

    raises OSError if the directory doesn't use a supported VCS.
    """
    for vcs_name in VCS_SUBCOMMANDS_BY_NAME:
        vcs_api = VCSAPI(name=vcs_name)
        if vcs_api.is_usable:
            return vcs_api
    raise OSError('No such directory .git/ or .hg/ ')


def assert_not_dirty(vcs_api, filepaths, allow_dirty):
    dirty_files = vcs_api.status(required_files=filepaths)
    if dirty_files:
        logger.warning(
            '{0} working directory is not clean. Uncomitted file(s):'.
            format(vcs_api.name))
        for dirty_file in dirty_files:
            logger.warning('    ' + dirty_file)
    if not allow_dirty and dirty_files:
        sys.exit(1)
    dirty_pattern_files = set(dirty_files) & filepaths
    if dirty_pattern_files:
        logger.error('Not commiting when pattern files are dirty:')
        for dirty_file in dirty_pattern_files:
            logger.warning('    ' + dirty_file)
        sys.exit(1)


def commit(cfg, vcs_api, filepaths, new_version, commit_message, tag_message):
    if cfg.commit:
        if cfg.pre_commit_hook:
            logger.info('Run pre-commit hook: {0}'.format(cfg.pre_commit_hook))
            hooks.run(cfg.pre_commit_hook, cfg.current_version, new_version)
        for filepath in filepaths:
            vcs_api.add(filepath)
        vcs_api.commit(commit_message)
        if cfg.post_commit_hook:
            logger.info('Run post-commit hook: {0}'.format(cfg.
                post_commit_hook))
            hooks.run(cfg.post_commit_hook, cfg.current_version, new_version)
    if cfg.commit and cfg.tag:
        vcs_api.tag(tag_name=new_version, tag_message=tag_message)
    if cfg.commit and cfg.push:
        if cfg.tag:
            vcs_api.push_tag(tag_name=new_version)
        else:
            vcs_api.push()


def get_tags(fetch, scope):
    try:
        vcs_api = get_vcs_api()
        logger.debug('vcs found: {0}'.format(vcs_api.name))
        if fetch:
            logger.info(
                'fetching tags from remote (to turn off use: -n / --no-fetch)')
            vcs_api.fetch()
        branch_scope = scope == config.TagScope.BRANCH
        if branch_scope:
            return vcs_api.ls_tags_branch()
        else:
            return vcs_api.ls_tags()
    except OSError:
        logger.debug('No vcs found')
        return []
