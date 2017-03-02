from setuptools import setup, find_packages

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