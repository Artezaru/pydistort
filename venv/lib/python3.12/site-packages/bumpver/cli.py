#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://github.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""cli module for BumpVer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import re
import sys
import typing as typ
import logging
import datetime as dt
import subprocess as sp
import click
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import colorama
str = getattr(builtins, 'unicode', str)
from . import vcs
from . import config
from . import rewrite
from . import version
from . import patterns
from . import regexfmt
from . import v1rewrite
from . import v1version
from . import v2rewrite
from . import v2version
from . import v1patterns
from . import v2patterns
try:
    import pretty_traceback
    pretty_traceback.install()
except ImportError:
    pass
click.disable_unicode_literals_warning = True
logger = logging.getLogger('bumpver.cli')
_VERBOSE = 0


def _configure_logging(verbose=0):
    global _VERBOSE
    _VERBOSE = verbose
    if verbose >= 2:
        log_format = (
            '%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-17s - %(message)s'
            )
        log_level = logging.DEBUG
    elif verbose == 1:
        log_format = '%(levelname)-7s - %(message)s'
        log_level = logging.INFO
    else:
        log_format = '%(levelname)-7s - %(message)s'
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format, datefmt=
        '%Y-%m-%dT%H:%M:%S')
    logger.debug('Logging configured.')


VALID_RELEASE_TAG_VALUES = 'alpha', 'beta', 'dev', 'rc', 'post', 'final'
_current_date = dt.date.today().isoformat()


def _validate_date(date, pin_date):
    if date and pin_date:
        logger.error(
            "Can only use either --pin-date or --date='{0}', not both.".
            format(date))
        sys.exit(1)
    if date is None:
        return None
    try:
        dt_val = dt.datetime.strptime(date, '%Y-%m-%d')
        return dt_val.date()
    except ValueError:
        logger.error(
            "Invalid parameter --date='{0}', must match format YYYY-0M-0D."
            .format(date), exc_info=True)
        sys.exit(1)


def _validate_release_tag(tag):
    if tag is None:
        return
    if tag in VALID_RELEASE_TAG_VALUES:
        return
    logger.error('Invalid argument --tag={0}'.format(tag))
    logger.error('Valid arguments are: {0}'.format(', '.join(
        VALID_RELEASE_TAG_VALUES)))
    sys.exit(1)


def _validate_flags(raw_pattern, major, minor, patch):
    if '{' in raw_pattern and '}' in raw_pattern:
        return
    valid = True
    if major and 'MAJOR' not in raw_pattern:
        logger.error("Flag --major is not applicable to pattern '{0}'".
            format(raw_pattern))
        valid = False
    if minor and 'MINOR' not in raw_pattern:
        logger.error("Flag --minor is not applicable to pattern '{0}'".
            format(raw_pattern))
        valid = False
    if patch and 'PATCH' not in raw_pattern:
        logger.error("Flag --patch is not applicable to pattern '{0}'".
            format(raw_pattern))
        valid = False
    if not valid:
        sys.exit(1)


def _log_no_change(subcmd, version_pattern):
    is_semver = ('{semver}' in version_pattern or 'MAJOR' in
        version_pattern and 'MAJOR' in version_pattern and 'PATCH' in
        version_pattern)
    if is_semver:
        logger.warning(
            'bumpver {0} [--major/--minor/--patch] required for use with SemVer.'
            .format(subcmd))
    else:
        available_flags = [('--' + part.lower()) for part in ['MAJOR',
            'MINOR', 'PATCH'] if part in version_pattern]
        if available_flags:
            available_flags_str = '/'.join(available_flags)
            logger.info('Perhaps try: bumpver {0} {1} '.format(subcmd,
                available_flags_str))


def _get_normalized_pattern(raw_pattern, version_pattern):
    is_version_pattern_required = ('{version}' in raw_pattern or 
        '{pep440_version}' in raw_pattern)
    if is_version_pattern_required and version_pattern is None:
        logger.error(
            'Argument --version-pattern=<PATTERN> is required for placeholders: {version}/{pep440_version}.'
            )
        sys.exit(1)
    elif version_pattern is None:
        _version_pattern = 'INVALID'
    else:
        _version_pattern = version_pattern
    if is_version_pattern_required:
        return v2patterns.normalize_pattern(_version_pattern, raw_pattern)
    else:
        return raw_pattern


verbose_option = click.option('-v', '--verbose', count=True, help=
    'Control log level. -vv for debug level.')
dry_option = click.option('-d', '--dry', default=False, is_flag=True, help=
    "Display diff of changes, don't rewrite files.")
allow_dirty_option = click.option('--allow-dirty', default=False, is_flag=
    True, help=
    'Commit even when working directory is has uncomitted changes. (WARNING: The commit will still be aborted if there are uncomitted to files with version strings.'
    )
ignore_vcs_tag_option = click.option('--ignore-vcs-tag', default=False,
    is_flag=True, help='Ignore VCS tag invariant and update version anyway.')
fetch_option = click.option('-f/-n', '--fetch/--no-fetch', is_flag=True,
    default=True, help='Sync tags from remote origin.')
env_option = click.option('-e', '--env', is_flag=True, default=False,
    hidden=True, help=
    'Print version state for use with shell scripts: eval $(bumpver show --env)'
    )
environ_option = click.option('--environ', is_flag=True, default=False,
    help=
    'Print version state for use with shell scripts: eval $(bumpver show --environ)'
    )


def version_options(function):
    decorators = [click.option('--major', is_flag=True, default=False, help
        ='Increment MAJOR component.'), click.option('-m', '--minor',
        is_flag=True, default=False, help='Increment MINOR component.'),
        click.option('-p', '--patch', is_flag=True, default=False, help=
        'Increment PATCH component.'), click.option('-t', '--tag', default=
        None, metavar='<NAME>', help=
        'Override release tag of current_version. Valid options are: {0}.'.
        format(', '.join(VALID_RELEASE_TAG_VALUES))), click.option(
        '--tag-num', is_flag=True, default=False, help=
        'Increment release tag number (rc1, rc2, rc3..).'), click.option(
        '--pin-increments', is_flag=True, default=False, help=
        'Leave the auto-increments INC0 and INC1 unchanged.'), click.option
        ('--pin-date', is_flag=True, default=False, help=
        'Leave date components unchanged.'), click.option('--date', default
        =None, metavar='<ISODATE>', help=
        'Set explicit date in format YYYY-0M-0D (e.g. {0}).'.format(
        _current_date)), click.option('--set-version', default=None,
        metavar='<VERSION>', help='Set version explicitly.')]
    decorated = function
    for decorator in decorators:
        decorated = decorator(decorated)
    return decorated


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(version='2024.1130')
@verbose_option
def cli(verbose=0):
    """Automatically update version strings in plaintext files."""
    if verbose:
        _configure_logging(verbose=max(_VERBOSE, verbose))


@cli.command()
@click.argument('old_version')
@click.argument('pattern')
@verbose_option
@version_options
def test(old_version, pattern, verbose=0, major=False, minor=False, patch=
    False, tag=None, tag_num=False, pin_increments=False, pin_date=False,
    date=None, set_version=None):
    """Increment a version number for demo purposes."""
    _configure_logging(verbose=max(_VERBOSE, verbose))
    _validate_release_tag(tag)
    raw_pattern = pattern
    _validate_flags(raw_pattern, major, minor, patch)
    maybe_date = _validate_date(date, pin_date)
    if set_version is None:
        new_version = incr_dispatch(old_version, raw_pattern=raw_pattern,
            major=major, minor=minor, patch=patch, tag=tag, tag_num=tag_num,
            pin_increments=pin_increments, pin_date=pin_date, maybe_date=
            maybe_date)
    else:
        new_version = set_version
    if new_version is None:
        _log_no_change('test', raw_pattern)
        sys.exit(1)
    if not _is_valid_version(raw_pattern, old_version, new_version):
        if set_version:
            logger.error("Invalid argument --set-version='{0}'".format(
                set_version))
        sys.exit(1)
    pep440_version = version.to_pep440(new_version)
    click.echo('New Version: {0}'.format(new_version))
    if new_version != pep440_version:
        click.echo('PEP440     : {0}'.format(pep440_version))


def _grep_text(pattern, text, color):
    all_lines = text.splitlines()
    for match in pattern.regexp.finditer(text):
        match_start, match_end = match.span()
        line_idx = text[:match_start].count('\n')
        line_start = text.rfind('\n', 0, match_start) + 1
        line_end = text.find('\n', match_end, -1)
        if color:
            matched_line = text[line_start:match_start
                ] + colorama.Style.BRIGHT + text[match_start:match_end
                ] + colorama.Style.RESET_ALL + text[match_end:line_end]
        else:
            matched_line = text[line_start:match_start] + text[match_start:
                match_end] + text[match_end:line_end]
        lines_offset = max(0, line_idx - 1) + 1
        lines = all_lines[line_idx - 1:line_idx + 2]
        if line_idx == 0:
            lines[0] = matched_line
        else:
            lines[1] = matched_line
        prefixed_lines = ['{0:>4}: {1}'.format(lines_offset + i, line) for 
            i, line in enumerate(lines)]
        yield '\n'.join(prefixed_lines)


def _grep(raw_pattern, file_ios, color):
    pattern = v2patterns.compile_pattern(raw_pattern)
    match_count = 0
    for file_io in file_ios:
        text = file_io.read()
        match_strs = list(_grep_text(pattern, text, color))
        if len(match_strs) > 0:
            if len(file_ios) > 1:
                print(file_io.name)
            for match_str in match_strs:
                print(match_str)
            print()
        match_count += len(match_strs)
    if match_count == 0:
        logger.error("Pattern not found: '{0}'".format(raw_pattern))
    if match_count == 0 or _VERBOSE:
        pyexpr_regex = regexfmt.pyexpr_regex(pattern.regexp.pattern)
        print('# ' + regexfmt.regex101_url(pattern.regexp.pattern))
        print(pyexpr_regex)
        print()
    if match_count == 0:
        sys.exit(1)


@cli.command()
@verbose_option
@click.option('--version-pattern', default=None, metavar='<PATTERN>', help=
    'Pattern to use for placeholders: {version}/{pep440_version}')
@click.argument('pattern')
@click.argument('files', nargs=-1, type=click.File('r'))
def grep(pattern, files, version_pattern=None, verbose=0):
    """Search file(s) for a version pattern."""
    verbose = max(_VERBOSE, verbose)
    _configure_logging(verbose)
    raw_pattern = pattern
    normalized_pattern = _get_normalized_pattern(raw_pattern, version_pattern)
    isatty = getattr(sys.stdout, 'isatty', lambda : False)
    if isatty():
        colorama.init()
        try:
            _grep(normalized_pattern, files, color=True)
        finally:
            colorama.deinit()
    else:
        _grep(normalized_pattern, files, color=False)


@cli.command()
@verbose_option
@ignore_vcs_tag_option
@fetch_option
@env_option
@environ_option
def show(verbose=0, ignore_vcs_tag=False, fetch=True, env=False, environ=False
    ):
    """Show current version of your project."""
    _configure_logging(verbose=max(_VERBOSE, verbose))
    _, cfg = config.init(project_path='.')
    if cfg is None:
        logger.error(
            "Could not parse configuration. Perhaps try 'bumpver init'.")
        sys.exit(1)
    if not ignore_vcs_tag:
        cfg = _update_cfg_from_vcs(cfg, fetch)
    if env:
        logger.warning('Depricated: -e/--env use --environ instead. ')
        logger.warning('    See https://github.com/mbarkhau/bumpver/issues/224'
            )
        version_info = v2version.parse_version_info(cfg.current_version,
            cfg.version_pattern)
        for key, val in version_info._asdict().items():
            click.echo('{0}={1}'.format(key.upper(), val if val else ''))
        click.echo('CURRENT_VERSION={0}'.format(cfg.current_version))
        click.echo('PEP440_VERSION={0}'.format(cfg.pep440_version))
    elif environ:
        version_info = v2version.parse_version_info(cfg.current_version,
            cfg.version_pattern)
        for key, val in version_info._asdict().items():
            click.echo('{0}={1}'.format(key.upper(), '' if val is False or 
                val is None else val))
        click.echo('CURRENT_VERSION={0}'.format(cfg.current_version))
        click.echo('PEP440_VERSION={0}'.format(cfg.pep440_version))
    else:
        click.echo('Current Version: {0}'.format(cfg.current_version))
        click.echo('PEP440         : {0}'.format(cfg.pep440_version))


def _colored_diff_lines(diff):
    for line in diff.splitlines():
        if line.startswith('+++') or line.startswith('---'):
            yield line
        elif line.startswith('+'):
            yield '\x1b[32m' + line + '\x1b[0m'
        elif line.startswith('-'):
            yield '\x1b[31m' + line + '\x1b[0m'
        elif line.startswith('@'):
            yield '\x1b[36m' + line + '\x1b[0m'
        else:
            yield line


def _v2_get_diff(cfg, new_version):
    old_vinfo = v2version.parse_version_info(cfg.current_version, cfg.
        version_pattern)
    new_vinfo = v2version.parse_version_info(new_version, cfg.version_pattern)
    return v2rewrite.diff(old_vinfo, new_vinfo, cfg.file_patterns)


def _v1_get_diff(cfg, new_version):
    old_vinfo = v1version.parse_version_info(cfg.current_version, cfg.
        version_pattern)
    new_vinfo = v1version.parse_version_info(new_version, cfg.version_pattern)
    return v1rewrite.diff(old_vinfo, new_vinfo, cfg.file_patterns)


def get_diff(cfg, new_version):
    if cfg.is_new_pattern:
        return _v2_get_diff(cfg, new_version)
    else:
        return _v1_get_diff(cfg, new_version)


def _print_diff_str(diff):
    colored_diff = '\n'.join(_colored_diff_lines(diff))
    if sys.stdout.isatty():
        click.echo(colored_diff)
    else:
        click.echo(diff)


def _print_diff(cfg, new_version):
    try:
        diff = get_diff(cfg, new_version)
        _print_diff_str(diff)
    except OSError as err:
        logger.error(str(err))
        sys.exit(1)
    except rewrite.NoPatternMatch as ex:
        logger.error(str(ex))
        sys.exit(1)


def _parse_version_tags(all_tags, version_pattern, is_new_pattern):
    version_parser = v2version if is_new_pattern else v1version
    return [tag for tag in all_tags if version_parser.is_valid(tag,
        version_pattern)]


def _is_valid_version(raw_pattern, old_version, new_version, unique=False):
    is_new_pattern = '{' not in raw_pattern and '}' not in raw_pattern
    try:
        if is_new_pattern:
            v2version.parse_version_info(new_version, raw_pattern)
        else:
            v1version.parse_version_info(new_version, raw_pattern)
    except version.PatternError:
        logger.error("Invalid version '{0}' for pattern '{1}'".format(
            new_version, raw_pattern))
        return False
    if version.parse_version(new_version) <= version.parse_version(old_version
        ):
        logger.error(
            'Invariant violated: New version must be greater than old version '
            )
        logger.error("  Failed Invariant: '{0}' > '{1}'".format(new_version,
            old_version))
        logger.error(
            "If the invariant is from vcs tags try '--ignore-vcs-tag' option.")
        return False
    if unique:
        all_tags = vcs.get_tags(fetch=False, scope=config.TagScope.GLOBAL)
        version_tags = _parse_version_tags(all_tags, raw_pattern,
            is_new_pattern)
        if new_version in version_tags:
            logger.error(
                'Invariant violated: New version must be unique accross all branches'
                )
            return False
    return True


def incr_dispatch(old_version, raw_pattern, **kwargs):
    major = kwargs.get('major', False)
    minor = kwargs.get('minor', False)
    patch = kwargs.get('patch', False)
    tag = kwargs.get('tag', None)
    tag_num = kwargs.get('tag_num', False)
    pin_increments = kwargs.get('pin_increments', False)
    pin_date = kwargs.get('pin_date', False)
    maybe_date = kwargs.get('maybe_date', None)
    v1_parts = list(v1patterns.PART_PATTERNS) + list(v1patterns.
        FULL_PART_FORMATS)
    has_v1_part = any('{' + part + '}' in raw_pattern for part in v1_parts)
    if _VERBOSE:
        if has_v1_part:
            pattern = v1patterns.compile_pattern(raw_pattern)
        else:
            pattern = v2patterns.compile_pattern(raw_pattern)
        logger.info('Using pattern ' + raw_pattern)
        logger.info('regex = ' + regexfmt.pyexpr_regex(pattern.regexp.pattern))
    if has_v1_part:
        return v1version.incr(old_version, raw_pattern=raw_pattern, major=
            major, minor=minor, patch=patch, tag=tag, tag_num=tag_num,
            pin_date=pin_date, maybe_date=maybe_date)
    else:
        return v2version.incr(old_version, raw_pattern=raw_pattern, major=
            major, minor=minor, patch=patch, tag=tag, tag_num=tag_num,
            pin_increments=pin_increments, pin_date=pin_date, maybe_date=
            maybe_date)


def _update(cfg, new_version, commit_message, tag_message, allow_dirty=False):
    vcs_api = None
    if cfg.commit:
        try:
            vcs_api = vcs.get_vcs_api()
        except OSError:
            logger.warning('Version Control System not found, skipping commit.'
                )
    filepaths = set(cfg.file_patterns.keys())
    if vcs_api:
        vcs.assert_not_dirty(vcs_api, filepaths, allow_dirty)
    try:
        if cfg.is_new_pattern:
            new_v2_vinfo = v2version.parse_version_info(new_version, cfg.
                version_pattern)
            v2rewrite.rewrite_files(cfg.file_patterns, new_v2_vinfo)
        else:
            new_v1_vinfo = v1version.parse_version_info(new_version, cfg.
                version_pattern)
            v1rewrite.rewrite_files(cfg.file_patterns, new_v1_vinfo)
    except rewrite.NoPatternMatch as ex:
        logger.error(str(ex))
        sys.exit(1)
    if vcs_api:
        vcs.commit(cfg, vcs_api, filepaths, new_version, commit_message,
            tag_message)


def _try_update(cfg, new_version, commit_message, tag_message, allow_dirty=
    False):
    try:
        _update(cfg, new_version, commit_message, tag_message, allow_dirty)
    except sp.CalledProcessError as ex:
        logger.error('Error running subcommand: {0}'.format(ex.cmd))
        if ex.stdout:
            sys.stdout.write(ex.stdout.decode('utf-8'))
        if ex.stderr:
            sys.stderr.write(ex.stderr.decode('utf-8'))
        sys.exit(1)


@cli.command()
@verbose_option
@dry_option
def init(verbose=0, dry=False):
    """Initialize [bumpver] configuration."""
    _configure_logging(verbose=max(_VERBOSE, verbose))
    ctx, cfg = config.init(project_path='.', cfg_missing_ok=True)
    if cfg:
        logger.error('Configuration already initialized in {0}'.format(ctx.
            config_rel_path))
        sys.exit(1)
    if dry:
        click.echo("Exiting because of '-d/--dry'. Would have written to {0}:"
            .format(ctx.config_rel_path))
        cfg_text = config.default_config(ctx)
        click.echo('\n    ' + '\n    '.join(cfg_text.splitlines()))
        sys.exit(0)
    config.write_content(ctx)


def get_latest_vcs_version_tag(cfg, fetch):
    all_tags = vcs.get_tags(fetch=fetch, scope=cfg.tag_scope)
    version_tags = _parse_version_tags(all_tags, cfg.version_pattern, cfg.
        is_new_pattern)
    if version_tags:
        version_tags.sort(key=version.parse_version, reverse=True)
        _debug_tags = ', '.join(version_tags[:3])
        logger.debug('found tags: {0} ... ({1} in total)'.format(
            _debug_tags, len(version_tags)))
        return version_tags[0]
    else:
        return None


def _update_cfg_from_vcs(cfg, fetch):
    latest_version_tag = get_latest_vcs_version_tag(cfg, fetch)
    if latest_version_tag is None:
        logger.debug('no vcs tags found')
        return cfg
    latest_version_pep440 = version.to_pep440(latest_version_tag)
    scope_str = '({0})'.format(cfg.tag_scope.value
        ) if not cfg.tag_scope == config.TagScope.DEFAULT else ''
    logger.info('Latest version from VCS tag: {0} {1}'.format(
        latest_version_tag, scope_str))
    if cfg.tag_scope == config.TagScope.DEFAULT:
        logger.info('Working dir version        : {0}'.format(cfg.
            current_version))
        if version.parse_version(latest_version_tag) <= version.parse_version(
            cfg.current_version):
            return cfg
    return cfg._replace(current_version=latest_version_tag, pep440_version=
        latest_version_pep440)


def _parse_vcs_options(cfg, commit=None, tag_commit=None, push=None,
    tag_scope=None, pre_commit_hook=None, post_commit_hook=None):
    if commit is False and tag_commit:
        raise ValueError(
            '--no-commit and --tag-commit cannot be used at the same time')
    if commit is False and push:
        raise ValueError(
            '--no-commit and --push cannot be used at the same time')
    if commit is not None:
        cfg = cfg._replace(commit=commit)
    if not cfg.commit and tag_commit:
        raise ValueError(
            '--tag-commit requires either --commit or commit=True in your config'
            )
    if not cfg.commit and push:
        raise ValueError(
            '--push requires either --commit or commit=True in your config')
    if tag_commit is not None:
        cfg = cfg._replace(tag=tag_commit)
    if push is not None:
        cfg = cfg._replace(push=push)
    if tag_scope is not None:
        cfg = cfg._replace(tag_scope=config.TagScope(tag_scope))
    if pre_commit_hook is not None:
        cfg = cfg._replace(pre_commit_hook=pre_commit_hook)
    if post_commit_hook is not None:
        cfg = cfg._replace(post_commit_hook=post_commit_hook)
    return cfg


def _sub_msg_template(message):
    return re.sub('\\b(OLD|NEW)\\b', '{\\1_VERSION}', message)


@cli.command()
@dry_option
@allow_dirty_option
@ignore_vcs_tag_option
@fetch_option
@verbose_option
@version_options
@click.option('-c', '--commit-message', default=None, metavar='<TMPL>',
    help='Set commit message template.')
@click.option('--tag-message', default=None, metavar='<TMPL>', help=
    'Set tag message template.')
@click.option('--commit/--no-commit', default=None, help=
    'Create a commit with all updated files.')
@click.option('--tag-commit/--no-tag-commit', default=None, help=
    'Tag the newly created commit.')
@click.option('--push/--no-push', default=None, help=
    'Push to the default remote.')
@click.option('--tag-scope', default=None, metavar=
    '[default|global|branch]', type=click.Choice([e.value for e in config.
    TagScope]), help='Tag scope for the current version.')
@click.option('--pre-commit-hook', type=click.Path(exists=True), metavar=
    '<PATH>', help='Custom script that runs before the commit step')
@click.option('--post-commit-hook', type=click.Path(exists=True), metavar=
    '<PATH>', help='Custom script that runs after the commit step is completed'
    )
def update(dry=False, allow_dirty=False, ignore_vcs_tag=False, fetch=True,
    verbose=0, major=False, minor=False, patch=False, tag=None, tag_num=
    False, pin_increments=False, pin_date=False, date=None, set_version=
    None, commit_message=None, tag_message=None, commit=None, tag_commit=
    None, push=None, tag_scope=None, pre_commit_hook=None, post_commit_hook
    =None):
    """Update project files with the incremented version string."""
    verbose = max(_VERBOSE, verbose)
    _configure_logging(verbose)
    _validate_release_tag(tag)
    maybe_date = _validate_date(date, pin_date)
    _, cfg = config.init(project_path='.')
    if cfg is None:
        logger.error('Could not parse configuration.')
        sys.exit(1)
    try:
        cfg = _parse_vcs_options(cfg, commit, tag_commit, push, tag_scope,
            pre_commit_hook, post_commit_hook)
    except ValueError as ex:
        logger.warning('Invalid argument: {0}'.format(ex))
        sys.exit(1)
    if not ignore_vcs_tag:
        cfg = _update_cfg_from_vcs(cfg, fetch)
    old_version = cfg.current_version
    if set_version is None:
        new_version = incr_dispatch(old_version, raw_pattern=cfg.
            version_pattern, major=major, minor=minor, patch=patch, tag=tag,
            tag_num=tag_num, pin_increments=pin_increments, pin_date=
            pin_date, maybe_date=maybe_date)
    else:
        new_version = set_version
    if new_version is None:
        _log_no_change('update', cfg.version_pattern)
        sys.exit(1)
    uniqueness_check = (cfg.tag_scope == config.TagScope.BRANCH or 
        set_version is not None)
    if not _is_valid_version(cfg.version_pattern, old_version, new_version,
        unique=uniqueness_check):
        if set_version:
            logger.error("Invalid argument --set-version='{0}'".format(
                set_version))
        sys.exit(1)
    logger.info('Old Version: {0}'.format(old_version))
    logger.info('New Version: {0}'.format(new_version))
    if dry or verbose >= 2:
        _print_diff(cfg, new_version)
    if commit_message is None:
        commit_msg_template = cfg.commit_message
    else:
        commit_msg_template = _sub_msg_template(commit_message)
    tag_msg_template = (cfg.tag_message if tag_message is None else
        _sub_msg_template(tag_message))
    tag_and_commit_message_kwargs = {'new_version': new_version,
        'old_version': old_version, 'NEW_VERSION': new_version,
        'OLD_VERSION': old_version, 'new_version_pep440': version.to_pep440
        (new_version), 'old_version_pep440': version.to_pep440(old_version)}
    try_commit_message = commit_msg_template.format(**
        tag_and_commit_message_kwargs)
    try_tag_message = tag_msg_template.format(**tag_and_commit_message_kwargs)
    if dry:
        return
    _try_update(cfg, new_version, try_commit_message, try_tag_message,
        allow_dirty)


if __name__ == '__main__':
    cli()
