# -*- coding: utf-8 -*-
import sys

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension  # lint:ok

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

cmd_class = {}

try:
    import setuptools.command.test
    class test_with_path(setuptools.command.test.test):
        def run(self, *p, **kw):
            setuptools.command.test.test.run(self, *p, **kw)
    cmd_class['test'] = test_with_path
except ImportError:
    test_with_path = None

if '--xml' in sys.argv:
    import unittest.runner
    import xmlrunner
    class XMLTestRunner(xmlrunner.XMLTestRunner):
        def __init__(self, *p, **kw):
            kw.setdefault('output', 'test-reports')
            xmlrunner.XMLTestRunner.__init__(self, *p, **kw)
    unittest.runner.TextTestRunner = xmlrunner.XMLTestRunner
    del sys.argv[sys.argv.index('--xml')]

if '--no-cython' in sys.argv:
    cythonize = None
    del sys.argv[sys.argv.index('--no-cython')]

VERSION = "0.2.0"

import re
import os.path
requires_re = re.compile(r'^([^<>=]*)((?:[<>=]{1,}[0-9.]*){1,})$')
with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
    readme = readme_file.read()
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as requirements_file:
    requirements = list(filter(bool, [ requires_re.sub(r'\1 (\2)', r.strip()) for r in requirements_file ]))

extra = {}

packages = [
      "sharedbuffers",
]

if cythonize is not None:
    include_dirs = os.environ.get('CYTHON_INCLUDE_DIRS','.').split(':')
    extension_modules = [
        Extension('sharedbuffers.mapped_struct', ['sharedbuffers/mapped_struct.py'],
            depends = ['sharedbuffers/mapped_struct.pxd']),
    ]
    extra['ext_modules'] = ext_modules = cythonize(extension_modules, include_path = include_dirs)

setup(
  name = "sharedbuffers",
  version = VERSION,
  description = "Shared-memory structured buffers",
  author = "Jampp",
  author_email = "klauss@jampp.com",
  maintainer = "Claudio Freire",
  maintainer_email = "klauss@jampp.com",
  url = "https://github.com/jampp/sharedbuffers/",
  #license = "?",
  long_description = readme,
  packages = packages,
  package_dir = {'sharedbuffers':'sharedbuffers'},
  
  tests_require = 'nose',
  test_suite = 'tests',
  license = 'BSD 3-Clause',

  cmdclass = cmd_class,

  requires = requirements,

  classifiers=[
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
  ],

  data_files = [
      ("", ["LICENSE"])
  ],

  zip_safe = False,
  **extra
)

