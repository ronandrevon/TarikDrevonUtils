import setuptools
import utils,os
#with open("latex/README.md", "r") as fh:
#    long_description = fh.read()
from utils import _version
setuptools.setup(
    name="TarikDrevonUtils",
    version=_version.version,
    author="Tarik Ronan Drevon",
    author_email="ronandrevon@gmail.com",
    description="some display utilities and others",
    long_description='', #long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'utils/materials':['materials*.pkl']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib','numpy','scipy','colorama','pandas','pillow','pytest','pytest-html'],#','PyQt5'],
)
