from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(
    name="safe_learning",
    version="0.0.1",
    author="Felix Berkenkamp",
    author_email="fberkenkamp@gmail.com",
    description=("An demonstration of how to create, document, and publish "
                  "to the cheese shop a5 pypi.org."),
    license="MIT",
    keywords="safe reinforcement learning Lyapunov",
    url="https://github.com/befelix/lyapunov-learning",
    packages=find_packages(exclude=['docs']),
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow>=1.0.0',
    ],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
