# -*- coding: utf-8 -*-
import sys
import os.path
import pkg_resources

# Earlier versions have an issue with Cython.Build and Extension subclassing
pkg_resources.require("setuptools>=20")

from setuptools import setup, Extension

no_setup_requires_arguments = set([
    '-h', '--help',
    '-n', '--dry-run',
    '-q', '--quiet',
    '-v', '--verbose',
    '-V', '--version',
    '--author',
    '--author-email',
    '--classifiers',
    '--contact',
    '--contact-email',
    '--description',
    '--egg-base',
    '--fullname',
    '--help-commands',
    '--keywords',
    '--licence',
    '--license',
    '--long-description',
    '--maintainer',
    '--maintainer-email',
    '--name',
    '--no-user-cfg',
    '--obsoletes',
    '--platforms',
    '--provides',
    '--requires',
    '--url',
    'clean',
    'egg_info',
    'register',
    'sdist',
    'upload',
])

cmd_class = {}
extra = {}

if set(sys.argv[1:]) <= no_setup_requires_arguments:
    cythonize = None
else:
    cythonize = True

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

setup_requires = ['Cython>=0.22,!=0.27.1,!=0.27.2,!=0.27.3']

if '--no-cython' in sys.argv:
    del sys.argv[sys.argv.index('--no-cython')]

    # As hilarious as it sounds, when built with --no-cython, we DO
    # need cython at runtime (to provide cython's shadow module), so
    # remove it from setup_requires to keep it in install_requires
    setup_requires = []
elif cythonize is None:
    from distutils.command.build import build
    class MissingCython(build):
        def run(self):
            pkg_resources.require('Cython')
    cmd_class.update({
        'build' : MissingCython,
        'install' : MissingCython,
        'test' : MissingCython,
    })
else:
    class lazy_modules(list):
        def initialize(self):
            if not getattr(self, 'initialized', False):
                self.initialized = True

                from Cython.Build import cythonize
                include_dirs = os.environ.get('CYTHON_INCLUDE_DIRS','.').split(':')
                extension_modules = [
                    Extension('sharedbuffers.mapped_struct', ['sharedbuffers/mapped_struct.py'],
                        depends = ['sharedbuffers/mapped_struct.pxd']),
                ]
                ext_modules = cythonize(extension_modules, include_path = include_dirs)
                self.extend(ext_modules)
        def __iter__(self):
            self.initialize()
            return list.__iter__(self)
        def __getitem__(self, key):
            self.initialize()
            return list.__getitem__(self, key)
        def __len__(self):
            self.initialize()
            return list.__len__(self)
        def __nonzero__(self):
            self.initialize()
            return len(self) > 0
    extra['ext_modules'] = lazy_modules()
    extra['setup_requires'] = setup_requires

VERSION = "0.4.9"

version_path = os.path.join(os.path.dirname(__file__), 'sharedbuffers', '_version.py')
if not os.path.exists(version_path):
    with open(version_path, "w") as version_file:
        pass
with open(version_path, "r+") as version_file:
    version_content = "__version__ = %r\n" % (VERSION,)
    if version_file.read() != version_content:
        version_file.seek(0)
        version_file.write(version_content)
        version_file.flush()
        version_file.truncate()
with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
    readme = readme_file.read()
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as requirements_file:
    requirements = list(filter(bool, [ r.strip() for r in requirements_file
        if r.strip() not in setup_requires ]))

packages = [
    "sharedbuffers",
]

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

    install_requires = requirements,

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

