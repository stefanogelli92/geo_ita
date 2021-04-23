from setuptools import setup, find_packages


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas", "twine", "geopandas", "geopy", "matplotlib>=3.1.1", "numpy", "descartes", "xlrd==1.2.0"]

setup(
    name="geo_ita",
    version="0.0.38",
    author="Stefano Gelli",
    author_email="stefano.mat92@gmail.com",
    description="A package for geo analysis for Italy",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/enelx-customer-business-analytics/geo_ita.git",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
            "geo_ita": ["data_sources/Anagrafica/*.*",
                         "data_sources/Comuni/Dimensioni/*.*",
                         "data_sources/Comuni/Popolazione/*.*",
                         "data_sources/Comuni/Shape/*.*",
                         "data_sources/Province/Shape/*.*",
                         "data_sources/Regioni/Shape/*.*"]
        },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)