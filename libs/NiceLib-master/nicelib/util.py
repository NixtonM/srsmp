# -*- coding: utf-8 -*-
# Copyright 2016-2018 Nate Bogdanowicz
from past.builtins import basestring

import sys
import os
import os.path
import logging
import warnings
from fnmatch import fnmatch
from ctypes.util import find_library
from contextlib import contextmanager

log = logging.getLogger(__name__)
is_64bit = sys.maxsize > 2**32

if sys.version_info < (3, 3):
    from chainmap import ChainMap
else:
    from collections import ChainMap


def select_platform_value(platform_dict):
    for platform_pattern, value in platform_dict.items():
        bitwidth = None
        if ':' in platform_pattern:
            platform_pattern, bitwidth = platform_pattern.split(':')
            if bitwidth not in ('32', '64'):
                raise ValueError("Only support 32 or 64 bits, got '{}'".format(bitwidth))
            pattern_is_64bit = bitwidth == '64'

        if bitwidth is None or (pattern_is_64bit == is_64bit):
            if fnmatch(sys.platform, platform_pattern):
                return value
    raise ValueError("No matching platform pattern found")


def to_tuple(value):
    """Convert value to a tuple, or if it is a string or dict, wrap it in a tuple"""
    return (value,) if isinstance(value, (basestring, dict)) else tuple(value)


# For Python 2 support
@contextmanager
def suppress(*exceptions):
    """Context manager to suppress specified exceptions"""
    try:
        yield
    except exceptions:
        pass


def handle_header_path(path, basedir):
    """Find the paths to the specified headers and verify they exist

    `path` may take a few forms. It may be just a string, in which case it specifies the absolute
    path to a header file.

    Otherwise, it must be in the form of a "platform" dict, which maps platform-specific strings
    to "header" dicts. Each "header" dict must contain a 'header' entry, which is a string or tuple
    of strings indicating header paths. Optionally you may include a 'path' entry, which specifies
    a list of directories where the headers may be located.
    """
    if isinstance(path, basestring):
        if os.path.exists(path):
            return path, None
        else:
            raise ValueError("Cannot find library header")

    header_tup = to_tuple(select_platform_value(path))
    for header_dict in header_tup:
        if 'header' not in header_dict:
            raise KeyError("Header dict must contain key 'header'")

        header_names = to_tuple(header_dict['header'])
        include_dirs = to_tuple(header_dict.get('path', ('.',)))
        log.info('Looking for {} in {}'.format(header_names, include_dirs))

        try:
            headers = [find_header(h, include_dirs, basedir) for h in header_names]
        except:
            continue

        if 'predef' in header_dict:
            try:
                predef_header = find_header(header_dict['predef'], include_dirs, basedir)
            except:
                continue
        else:
            predef_header = None

        return headers, predef_header
    raise ValueError("Could not find library header")


def find_header(header_name, include_dirs, basedir):
    """Resolve the path of a header_name

    Supports inclusion of environment variables in header and dir names. If `header_name` is a
    relative path (e.g. simply a filename), it is searched for relative to each of the paths in
    `include_dirs`, otherwise `include_dirs` is ignored and the absolute path is used directly.
    Raises an exception if the header cannot be found.
    """
    try:
        header_name = header_name.format(**os.environ)
    except KeyError as e:
        warnings.warn("os.environ does not provide key '{}'".format(e.args[0]))

    if os.path.isabs(header_name):
        if os.path.exists(header_name):
            return header_name
    else:
        for include_dir in include_dirs:
            try:
                include_dir = include_dir.format(**os.environ)
            except KeyError as e:
                warnings.warn("os.environ does not provide key '{}'".format(e.args[0]))

            if not os.path.isabs(include_dir):
                include_dir = os.path.join(basedir, include_dir)

            path = os.path.join(include_dir, header_name)
            if os.path.exists(path):
                return path

    raise Exception("Cannot find header '{}'".format(header_name))


def handle_lib_name(lib_name, basedir):
    """Find the path to the library

    `lib_name` can be specified directly as a string (or sequence of strings), or within a
    "platform" dict. If multiple libs are specified, the first one found will be returned.
    """
    if isinstance(lib_name, dict):
        lib_names = to_tuple(select_platform_value(lib_name))
    else:
        lib_names = to_tuple(lib_name)

    # First try local directory
    for try_name in lib_names:
        if os.path.isabs(try_name):
            path = try_name
        else:
            path = os.path.join(basedir, try_name)

        if os.path.exists(path):
            return path

    # Then try system directory
    for try_name in lib_names:
        path = find_library(try_name)
        if path:
            return path

    raise ValueError("Cannot find library '{}'".format(lib_names))
