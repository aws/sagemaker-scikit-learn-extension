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
General tests for all estimators in sagemaker-sklearn-extension.
"""
import pytest

from sklearn.utils.estimator_checks import check_estimator

from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.impute import RobustMissingIndicator
from sagemaker_sklearn_extension.preprocessing import LogExtremeValuesTransformer
from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.preprocessing import QuadraticFeatures
from sagemaker_sklearn_extension.preprocessing import QuantileExtremeValuesTransformer
from sagemaker_sklearn_extension.preprocessing import RemoveConstantColumnsTransformer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder


@pytest.mark.parametrize(
    "Estimator",
    [
        DateTimeVectorizer,
        LogExtremeValuesTransformer,
        MultiColumnTfidfVectorizer,
        NALabelEncoder,
        QuadraticFeatures,
        QuantileExtremeValuesTransformer,
        RobustImputer,
        RemoveConstantColumnsTransformer,
        RobustLabelEncoder,
        RobustMissingIndicator,
        RobustStandardScaler,
        ThresholdOneHotEncoder,
    ],
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
