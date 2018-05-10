from setuptools import setup, find_packages


setup(
    name='vanillaML',
    version="0.0.1",
    packages=find_packages(),
    install_requires=['numpy'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Vanilla implementations of machine learning algorithms with Numpy.",
    license="MIT",
    keywords="vanilla, ML",
    url="https://github.com/yngtodd/vanillaML",
)
