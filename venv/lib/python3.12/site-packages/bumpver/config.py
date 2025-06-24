# -*- coding: utf-8 -*-
# This file is part of the bumpver project
# https://gitlab.com/mbarkhau/bumpver
#
# Copyright (c) 2018-2024 Manuel Barkhau (mbarkhau@gmail.com) - MIT License
# SPDX-License-Identifier: MIT
"""Parse bumpver.toml, setup.cfg or pyproject.toml files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import enum
import typing as typ
import logging
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
import toml
str = getattr(builtins, 'unicode', str)
from bumpver import pathlib as pl
from . import utils
from . import version
from . import v1version
from . import v2version
from . import v1patterns
from . import v2patterns
from .patterns import Pattern
logger = logging.getLogger('bumpver.config')
RawPatterns = typ.List[str]
RawPatternsByFile = typ.Dict[str, RawPatterns]
FileRawPatternsItem = typ.Tuple[str, RawPatterns]
PatternsByFile = typ.Dict[str, typ.List[Pattern]]
FilePatternsItem = typ.Tuple[str, typ.List[Pattern]]
SUPPORTED_CONFIGS = ['setup.cfg', 'pyproject.toml', 'pycalver.toml',
    'bumpver.toml', '.bumpver.toml']


class TagScope(str, enum.Enum):
    DEFAULT = 'default'
    GLOBAL = 'global'
    BRANCH = 'branch'


DEFAULT_TAG_SCOPE = TagScope.DEFAULT
DEFAULT_COMMIT_MESSAGE = 'bump version to {new_version}'
DEFAULT_TAG_MESSAGE = '{new_version}'
ProjectContext = typ.NamedTuple('ProjectContext', [('path', pl.Path), (
    'config_filepath', pl.Path), ('config_rel_path', str), ('config_format',
    str), ('vcs_type', typ.Optional[str])])


def _pick_config_filepath(path):
    config_candidates = [path / 'pycalver.toml', path / 'bumpver.toml', 
        path / '.bumpver.toml', path / 'pyproject.toml', path / 'setup.cfg']
    for config_filepath in config_candidates:
        if config_filepath.exists():
            with config_filepath.open(mode='rb') as fobj:
                data = fobj.read()
            has_bumpver_section = (b'bumpver]' in data or b'pycalver]' in data
                ) and b'current_version' in data
            if has_bumpver_section:
                return config_filepath
    for config_filepath in config_candidates:
        if config_filepath.exists():
            return config_filepath
    return path / 'bumpver.toml'


def _parse_config_and_format(path):
    config_filepath = _pick_config_filepath(path)
    if config_filepath.is_absolute():
        config_rel_path = str(config_filepath.relative_to(path.absolute()))
    else:
        config_rel_path = str(config_filepath)
        config_filepath = pl.Path.cwd() / config_filepath
    config_format = config_filepath.suffix[1:]
    return config_filepath, config_rel_path, config_format


def init_project_ctx(project_path='.'):
    """Initialize ProjectContext from a path."""
    if isinstance(project_path, pl.Path):
        path = project_path
    elif project_path is None:
        path = pl.Path('.')
    else:
        path = pl.Path(str(project_path))
    config_filepath, config_rel_path, config_format = _parse_config_and_format(
        path)
    vcs_type = None
    if (path / '.git').exists():
        vcs_type = 'git'
    elif (path / '.hg').exists():
        vcs_type = 'hg'
    else:
        vcs_type = None
    return ProjectContext(path, config_filepath, config_rel_path,
        config_format, vcs_type)


RawConfig = typ.Dict[str, typ.Any]
MaybeRawConfig = typ.Optional[RawConfig]
Config = typ.NamedTuple('Config', [('current_version', str), (
    'version_pattern', str), ('pep440_version', str), ('commit_message',
    str), ('tag_message', str), ('tag_scope', TagScope), ('pre_commit_hook',
    str), ('post_commit_hook', str), ('commit', bool), ('tag', bool), (
    'push', bool), ('is_new_pattern', bool), ('file_patterns', PatternsByFile)]
    )
MaybeConfig = typ.Optional[Config]


def _debug_str(cfg):
    cfg_str_parts = ['Config Parsed: Config(',
        """
    current_version='{0}',""".format(cfg.current_version),
        """
    version_pattern='{0}',""".format(cfg.version_pattern),
        """
    pep440_version='{0}',""".format(cfg.pep440_version),
        """
    commit_message='{0}',""".format(cfg.commit_message),
        """
    tag_message='{0}',""".format(cfg.tag_message),
        """
    tag_scope='{0}',""".format(cfg.tag_scope.value),
        """
    pre_commit_hook={0},""".format(cfg.pre_commit_hook),
        """
    post_commit_hook={0},""".format(cfg.post_commit_hook),
        '\n    commit={0},'.format(cfg.commit), '\n    tag={0},'.format(cfg
        .tag), '\n    push={0},'.format(cfg.push),
        """
    is_new_pattern={0},""".format(cfg.is_new_pattern),
        """
    file_patterns={"""]
    for filepath, patterns in sorted(cfg.file_patterns.items()):
        for pattern in patterns:
            cfg_str_parts.append("\n        '{0}': '{1}',".format(filepath,
                pattern.raw_pattern))
    cfg_str_parts += ['\n    }\n)']
    return ''.join(cfg_str_parts)


def _parse_cfg_file_patterns(cfg_parser):
    file_pattern_items = None
    if cfg_parser.has_section('pycalver:file_patterns'):
        file_pattern_items = cfg_parser.items('pycalver:file_patterns')
    elif cfg_parser.has_section('bumpver:file_patterns'):
        file_pattern_items = cfg_parser.items('bumpver:file_patterns')
    else:
        return
    for filepath, patterns_str in file_pattern_items:
        maybe_patterns = (line.strip() for line in patterns_str.splitlines())
        patterns = [p for p in maybe_patterns if p]
        yield filepath, patterns


class _ConfigParser(configparser.RawConfigParser):
    """Custom parser, simply to override optionxform behaviour."""

    def optionxform(self, optionstr):
        """Non-xforming (ie. uppercase preserving) override.

        This is important because our option names are actually
        filenames, so case sensitivity is relevant. The default
        behaviour is to do optionstr.lower()
        """
        return optionstr


OptionVal = typ.Union[str, bool, None]
BOOL_OPTIONS = {'commit': False, 'tag': None, 'push': None}


def _parse_cfg(cfg_buffer):
    cfg_parser = _ConfigParser()
    if hasattr(cfg_parser, 'read_file'):
        cfg_parser.read_file(cfg_buffer)
    else:
        cfg_parser.readfp(cfg_buffer)
    raw_cfg = None
    if cfg_parser.has_section('pycalver'):
        raw_cfg = dict(cfg_parser.items('pycalver'))
    elif cfg_parser.has_section('bumpver'):
        raw_cfg = dict(cfg_parser.items('bumpver'))
    else:
        logger.warning("Perhaps try 'bumpver init'.")
        raise ValueError('Missing [bumpver] section.')
    for option, default_val in BOOL_OPTIONS.items():
        val = raw_cfg.get(option, default_val)
        if isinstance(val, (bytes, str)):
            val = val.lower() in ('yes', 'true', '1', 'on')
        raw_cfg[option] = val
    raw_cfg['file_patterns'] = dict(_parse_cfg_file_patterns(cfg_parser))
    _set_raw_config_defaults(raw_cfg)
    return raw_cfg


def _parse_toml(cfg_buffer):
    raw_full_cfg = toml.load(cfg_buffer)
    raw_cfg = None
    if 'tool' in raw_full_cfg and 'bumpver' in raw_full_cfg['tool']:
        raw_cfg = raw_full_cfg['tool']['bumpver']
    elif 'bumpver' in raw_full_cfg:
        raw_cfg = raw_full_cfg['bumpver']
    elif 'pycalver' in raw_full_cfg:
        raw_cfg = raw_full_cfg['pycalver']
    else:
        raw_cfg = {}
    for option, default_val in BOOL_OPTIONS.items():
        raw_cfg[option] = raw_cfg.get(option, default_val)
    _set_raw_config_defaults(raw_cfg)
    return raw_cfg


def _iter_glob_expanded_file_patterns(raw_patterns_by_file):
    for filepath_glob, raw_patterns in raw_patterns_by_file.items():
        filepaths = list(pl.Path().glob(filepath_glob))
        if filepaths:
            for filepath in filepaths:
                yield str(filepath), raw_patterns
        else:
            logger.warning('Invalid config, no such file: {0}'.format(
                filepath_glob))
            yield filepath_glob, raw_patterns


def _compile_v1_file_patterns(raw_cfg):
    """Create inernal/compiled representation of the file_patterns config field.

    The result the same, regardless of the config format.
    """
    version_pattern = raw_cfg['version_pattern']
    raw_patterns_by_file = raw_cfg['file_patterns']
    for filepath, raw_patterns in _iter_glob_expanded_file_patterns(
        raw_patterns_by_file):
        compiled_patterns = v1patterns.compile_patterns(version_pattern,
            raw_patterns)
        yield filepath, compiled_patterns


def _compile_v2_file_patterns(raw_cfg):
    """Create inernal/compiled representation of the file_patterns config field.

    The result the same, regardless of the config format.
    """
    version_pattern = raw_cfg['version_pattern']
    raw_patterns_by_file = raw_cfg['file_patterns']
    for filepath, raw_patterns in _iter_glob_expanded_file_patterns(
        raw_patterns_by_file):
        for raw_pattern in raw_patterns:
            if raw_pattern.startswith('['):
                errmsg = 'Invalid pattern {0} for {1}. '.format(raw_pattern,
                    filepath) + "Character not valid in this position '[' "
                raise ValueError(errmsg)
            try:
                v2patterns.compile_pattern(version_pattern, raw_pattern)
            except re.error:
                logger.warning('Invalid patterns for {0} ({1})'.format(
                    filepath, raw_pattern))
                raise
        compiled_patterns = v2patterns.compile_patterns(version_pattern,
            raw_patterns)
        yield filepath, compiled_patterns


def _compile_file_patterns(raw_cfg, is_new_pattern):
    if is_new_pattern:
        _file_pattern_items = _compile_v2_file_patterns(raw_cfg)
    else:
        _file_pattern_items = _compile_v1_file_patterns(raw_cfg)
    file_patterns = {}
    for path, patterns in _file_pattern_items:
        if path in file_patterns:
            file_patterns[path].extend(patterns)
        else:
            file_patterns[path] = patterns
    return file_patterns


def _validate_version_with_pattern(current_version, version_pattern,
    is_new_pattern):
    """Provoke ValueError if version_pattern and current_version are not compatible."""
    try:
        if is_new_pattern:
            v2version.parse_version_info(current_version, version_pattern)
        else:
            v1version.parse_version_info(current_version, version_pattern)
    except version.PatternError:
        errmsg = (
            "Invalid configuration. current_version='{0}' is invalid for version_pattern='{1}'"
            .format(current_version, version_pattern))
        raise ValueError(errmsg)
    if is_new_pattern:
        invalid_chars = re.search('([\\s]+)', version_pattern)
        if invalid_chars:
            errmsg = ('Invalid character(s) \'{0}\' in version_pattern = "{1}"'
                .format(invalid_chars.group(1), version_pattern))
            raise ValueError(errmsg)
        if not v2version.is_valid_week_pattern(version_pattern):
            errmsg = 'Invalid week number pattern: {0}'.format(version_pattern)
            raise ValueError(errmsg)


def _parse_cfg_strings(raw_cfg, key, default):
    if key in raw_cfg:
        raw_cfg[key] = raw_cfg[key].strip('\'" ')
    return raw_cfg.get(key, default)


def _parse_config(raw_cfg):
    """Parse configuration which was loaded from an .ini/.cfg or .toml file."""
    commit_message = raw_cfg.get('commit_message', DEFAULT_COMMIT_MESSAGE)
    commit_message = raw_cfg['commit_message'] = commit_message.strip('\'" ')
    tag_message = raw_cfg.get('tag_message', DEFAULT_TAG_MESSAGE)
    tag_message = raw_cfg['tag_message'] = tag_message.strip('\'" ')
    current_version = raw_cfg['current_version']
    current_version = raw_cfg['current_version'] = current_version.strip('\'" '
        )
    version_pattern = raw_cfg['version_pattern']
    version_pattern = raw_cfg['version_pattern'] = version_pattern.strip('\'" '
        )
    is_new_pattern = '{' not in version_pattern and '}' not in version_pattern
    _validate_version_with_pattern(current_version, version_pattern,
        is_new_pattern)
    pep440_version = version.to_pep440(current_version)
    file_patterns = _compile_file_patterns(raw_cfg, is_new_pattern)
    tag_scope = TagScope(_parse_cfg_strings(raw_cfg, 'tag_scope',
        DEFAULT_TAG_SCOPE))
    pre_commit_hook = _parse_cfg_strings(raw_cfg, 'pre_commit_hook', '')
    post_commit_hook = _parse_cfg_strings(raw_cfg, 'post_commit_hook', '')
    commit = raw_cfg['commit']
    tag = raw_cfg['tag']
    push = raw_cfg['push']
    if tag is None:
        tag = raw_cfg['tag'] = False
    if push is None:
        push = raw_cfg['push'] = False
    if tag and not commit:
        raise ValueError('commit=True required if tag=True')
    if push and not commit:
        raise ValueError('commit=True required if push=True')
    if tag_scope not in list(TagScope):
        raise ValueError('Invalid value for tag_scope')
    if pre_commit_hook and not pl.Path(pre_commit_hook).exists():
        errmsg = (
            "Invalid value for pre_commit_hook: Path '{0}' does not exist".
            format(pre_commit_hook))
        raise ValueError(errmsg)
    if post_commit_hook and not pl.Path(post_commit_hook).exists():
        errmsg = (
            "Invalid value for post_commit_hook: Path '{0}' does not exist"
            .format(post_commit_hook))
        raise ValueError(errmsg)
    cfg = Config(current_version=current_version, version_pattern=
        version_pattern, pep440_version=pep440_version, commit_message=
        commit_message, tag_message=tag_message, tag_scope=tag_scope,
        pre_commit_hook=pre_commit_hook, post_commit_hook=post_commit_hook,
        commit=commit, tag=tag, push=push, is_new_pattern=is_new_pattern,
        file_patterns=file_patterns)
    logger.debug(_debug_str(cfg))
    return cfg


def _parse_current_version_default_pattern(raw_cfg, raw_cfg_text):
    is_config_section = False
    for line in raw_cfg_text.splitlines():
        if is_config_section and line.startswith('current_version'):
            current_version = raw_cfg['current_version']
            version_pattern = raw_cfg['version_pattern']
            return line.replace(current_version, version_pattern)
        if line.strip() == '[pycalver]':
            is_config_section = True
        elif line.strip() == '[bumpver]':
            is_config_section = True
        elif line.strip() == '[tool.bumpver]':
            is_config_section = True
        elif line and line[0] == '[' and line[-1] == ']':
            is_config_section = False
    raise ValueError("Could not parse 'current_version'")


def _set_raw_config_defaults(raw_cfg):
    if 'version_pattern' not in raw_cfg:
        raise TypeError('Missing version_pattern')
    elif not isinstance(raw_cfg['version_pattern'], str):
        err = 'Invalid type for version_pattern = {0}'.format(raw_cfg[
            'version_pattern'])
        raise TypeError(err)
    if 'current_version' not in raw_cfg:
        raise ValueError("Missing 'current_version' configuration")
    elif not isinstance(raw_cfg['current_version'], str):
        err = 'Invalid type for current_version = {0}'.format(raw_cfg[
            'current_version'])
        raise TypeError(err)
    if 'file_patterns' not in raw_cfg:
        raw_cfg['file_patterns'] = {}


def _parse_raw_config(ctx):
    with ctx.config_filepath.open(mode='rt', encoding='utf-8') as fobj:
        if ctx.config_format == 'toml':
            raw_cfg = _parse_toml(fobj)
        elif ctx.config_format == 'cfg':
            raw_cfg = _parse_cfg(fobj)
        else:
            err_msg = (
                "Invalid config_format='{0}'.Supported formats are 'setup.cfg' and 'pyproject.toml'"
                .format(ctx.config_format))
            raise RuntimeError(err_msg)
    if ctx.config_rel_path not in raw_cfg['file_patterns']:
        with ctx.config_filepath.open(mode='rt', encoding='utf-8') as fobj:
            raw_cfg_text = fobj.read()
        raw_version_pattern = _parse_current_version_default_pattern(raw_cfg,
            raw_cfg_text)
        raw_cfg['file_patterns'][ctx.config_rel_path] = [raw_version_pattern]
    return raw_cfg


def parse(ctx, cfg_missing_ok=False):
    """Parse config file if available."""
    if ctx.config_filepath.exists():
        try:
            raw_cfg = _parse_raw_config(ctx)
            return _parse_config(raw_cfg)
        except (TypeError, ValueError) as ex:
            logger.warning("Couldn't parse {0}: {1}".format(ctx.
                config_rel_path, str(ex)))
            return None
    elif cfg_missing_ok:
        return None
    else:
        logger.warning('File not found: {0}'.format(ctx.config_rel_path))
        return None


def init(project_path='.', cfg_missing_ok=False):
    ctx = init_project_ctx(project_path)
    cfg = parse(ctx, cfg_missing_ok)
    return ctx, cfg


DEFAULT_CONFIGPARSER_BASE_TMPL = (
    """
[bumpver]
current_version = "{initial_version}"
version_pattern = "YYYY.BUILD[-TAG]"
commit_message = "bump version {{old_version}} -> {{new_version}}"
tag_message = "{{new_version}}"
tag_scope = "{default_tag_scope}"
pre_commit_hook = ""
post_commit_hook = ""
commit = True
tag = True
push = True

[bumpver:file_patterns]
"""
    .lstrip())
DEFAULT_CONFIGPARSER_SETUP_CFG_STR = (
    """
setup.cfg =
    current_version = "{version}\"
""".lstrip())
DEFAULT_CONFIGPARSER_SETUP_PY_STR = (
    """
setup.py =
    "{version}"
    "{pep440_version}\"
""".lstrip())
DEFAULT_CONFIGPARSER_README_RST_STR = (
    """
README.rst =
    {version}
    {pep440_version}
""".lstrip())
DEFAULT_CONFIGPARSER_README_MD_STR = (
    """
README.md =
    {version}
    {pep440_version}
""".lstrip())
DEFAULT_PYPROJECT_TOML_BASE_TMPL = (
    """
[tool.bumpver]
current_version = "{initial_version}"
version_pattern = "YYYY.BUILD[-TAG]"
commit_message = "bump version {{old_version}} -> {{new_version}}"
tag_message = "{{new_version}}"
tag_scope = "{default_tag_scope}"
pre_commit_hook = ""
post_commit_hook = ""
commit = true
tag = true
push = true

[tool.bumpver.file_patterns]
"""
    .lstrip())
DEFAULT_BUMPVER_TOML_BASE_TMPL = (
    """
[bumpver]
current_version = "{initial_version}"
version_pattern = "YYYY.BUILD[-TAG]"
commit_message = "bump version {{old_version}} -> {{new_version}}"
tag_message = "{{new_version}}"
tag_scope = "{default_tag_scope}"
pre_commit_hook = ""
post_commit_hook = ""
commit = true
tag = true
push = true

[bumpver.file_patterns]
"""
    .lstrip())
DEFAULT_TOML_PYCALVER_STR = (
    """
"pycalver.toml" = [
    'current_version = "{version}"',
]
""".lstrip()
    )
DEFAULT_TOML_BUMPVER_STR = (
    """
"bumpver.toml" = [
    'current_version = "{version}"',
]
""".lstrip())
DEFAULT_TOML_DOT_BUMPVER_STR = (
    """
".bumpver.toml" = [
    'current_version = "{version}"',
]
""".lstrip()
    )
DEFAULT_TOML_PYPROJECT_STR = (
    """
"pyproject.toml" = [
    'current_version = "{version}"',
]
""".
    lstrip())
DEFAULT_TOML_SETUP_PY_STR = (
    """
"setup.py" = [
    "{version}",
    "{pep440_version}",
]
""".lstrip())
DEFAULT_TOML_README_RST_STR = (
    """
"README.rst" = [
    "{version}",
    "{pep440_version}",
]
""".
    lstrip())
DEFAULT_TOML_README_MD_STR = (
    """
"README.md" = [
    "{version}",
    "{pep440_version}",
]
""".lstrip()
    )


def _initial_version():
    return utils.now().strftime('%Y.1001-alpha')


def _initial_version_pep440():
    return utils.now().strftime('%Y.1001a0')


def default_config(ctx):
    """Generate initial default config."""
    fmt = ctx.config_format
    if fmt == 'cfg':
        base_tmpl = DEFAULT_CONFIGPARSER_BASE_TMPL
        default_pattern_strs_by_filename = {'setup.cfg':
            DEFAULT_CONFIGPARSER_SETUP_CFG_STR, 'setup.py':
            DEFAULT_CONFIGPARSER_SETUP_PY_STR, 'README.rst':
            DEFAULT_CONFIGPARSER_README_RST_STR, 'README.md':
            DEFAULT_CONFIGPARSER_README_MD_STR}
    elif fmt == 'toml':
        if ctx.config_filepath.name == 'pyproject.toml':
            base_tmpl = DEFAULT_PYPROJECT_TOML_BASE_TMPL
        else:
            base_tmpl = DEFAULT_BUMPVER_TOML_BASE_TMPL
        default_pattern_strs_by_filename = {'pyproject.toml':
            DEFAULT_TOML_PYPROJECT_STR, 'pycalver.toml':
            DEFAULT_TOML_PYCALVER_STR, 'bumpver.toml':
            DEFAULT_TOML_BUMPVER_STR, '.bumpver.toml':
            DEFAULT_TOML_DOT_BUMPVER_STR, 'setup.py':
            DEFAULT_TOML_SETUP_PY_STR, 'README.rst':
            DEFAULT_TOML_README_RST_STR, 'README.md':
            DEFAULT_TOML_README_MD_STR}
    else:
        raise ValueError(
            "Invalid config_format='{0}', must be either 'toml' or 'cfg'.".
            format(fmt))
    cfg_str = base_tmpl.format(initial_version=_initial_version(),
        default_tag_scope=DEFAULT_TAG_SCOPE.value)
    for filename, default_str in default_pattern_strs_by_filename.items():
        if (ctx.path / filename).exists():
            cfg_str += default_str
    has_config_file = any((ctx.path / fn).exists() for fn in SUPPORTED_CONFIGS)
    if not has_config_file:
        if ctx.config_format == 'cfg':
            cfg_str += DEFAULT_CONFIGPARSER_SETUP_CFG_STR
        if ctx.config_format == 'toml':
            cfg_str += DEFAULT_TOML_BUMPVER_STR
    cfg_str += '\n'
    return cfg_str


def write_content(ctx):
    """Update project config file with initial default config."""
    fobj = None
    cfg_content = default_config(ctx)
    if ctx.config_filepath.exists():
        cfg_content = '\n' + cfg_content
    with ctx.config_filepath.open(mode='at', encoding='utf-8') as fobj:
        fobj.write(cfg_content)
    print('Updated {0}'.format(ctx.config_rel_path))
