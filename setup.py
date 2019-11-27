# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#      http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


setup(
    name="sagemaker-scikit-learn-extension",
    version=read_version(),
    description="Open source library extension of scikit-learn for Amazon SageMaker.",
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    long_description=read("README.rst"),
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-scikit-learn-extension/",
    license="Apache License 2.0",
    keywords="ML Amazon AWS AI SKLearn Scikit-Learn",
    classifiers=["Development Status :: 4 - Beta", "License :: OSI Approved :: Apache Software License"],
    extras_require={"test": ["tox", "tox-conda", "pytest", "coverage"]},
)
