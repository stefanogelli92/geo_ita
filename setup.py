from setuptools import setup, find_packages


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas", "twine", "geopandas", "geopy",
                "matplotlib>=3.1.1", "numpy", "descartes",
                "xlrd", "openpyxl", "unidecode", "pyproj", "rtree",
                "bokeh", "requests", "beautifulsoup4", "google", "scipy",
                "google-api-python-client", "valdec", "pydantic"]

setup(
    name="geo_ita",
    version="0.0.2",
    author="Stefano Gelli",
    author_email="stefano.mat92@gmail.com",
    description="A package for geo analysis for Italy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanogelli92/geo_ita.git",
    packages=find_packages(exclude=("geo_ita.src")),
    install_requires=requirements,
    include_package_data=True,
    package_data={
            "geo_ita": ["data_sources/*.*"]
        },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)