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

"""
The :mod:`sagemaker_sklearn_extension.feature_extraction` module deals
with feature extraction from raw data. It currently includes estimators
to extract features from text. This module is based on the
:mod:`sklearn.feature_extraction` module.
"""

from . import date_time
from . import text

__all__ = ["date_time", "text"]
