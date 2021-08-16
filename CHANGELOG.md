# Changelog

## v2.3.0 (2021-08-16)

### Features

 * transformers for time series

## v2.2.1 (2021-05-21)

### Bug fixes and other changes

 * Datetime fix

## v2.2.0 (2021-04-13)

### Features

 * taei contrib library

### Bug fixes and other changes

 * broken tests and dependencies

## v2.1.0 (2020-10-21)

### Features

 * adds threshold and max_categories parameter to RobustOrdinalEncoder
 * Add weight of evidence encoder

### Bug fixes and other changes

 * use named functions instead of lambdas in DateTimeDefintions because of pickle

## v2.0.0 (2020-08-13)

### Breaking changes

 * update sklearn dependency version to 0.23 and mlio version to 0.5

### Features

 * OrdinalEncoder can output np.nan instead of n for unseen values

### Bug fixes and other changes

 * minor performance optimizations and refactoring

## v1.2.0 (2020-07-29)

### Features

 * adds a `get_classes` method to `RobustLabelEncoder`

## v1.1.1 (2020-07-21)

### Bug fixes and other changes

 * Merge pull request #18 from ipanepen/rle-bug
 * test data reading when n_rows = 1 mod batch_size
 * bug fix: makes fit_transform behavior consistent with fit and transform
 * fix a minor bug in OneHotEncoder by by overloading the buggy method in ThresholdOneHotEncoder and fixing it

## v1.1.0 (2020-02-24)

### Features

 * dummy feature commit for RobustOrdinalEncoder & add badges to README

### Bug fixes and other changes

 * libprotobuf==3.11.4 is not backwards compatible, specify tox version for testing
 * Merge pull request #11 from ipanepen/master
 * fix for MemoryError in ThresholdOneHotEncoder
 * Adding RobustOrdinalEncoder
 * Specify mlio version 0.2.7

## v1.0.0 (2019-12-03)

### Bug fixes and other changes

 * update to 1.0.0, fix buildspec
 * update ci deployment credentials
 * Merge pull request #4 from wiltonwu/master
 * update documentation, remove CHANGELOG.md for 0.1.0 deployment, add date_time module
 * Merge pull request #2 from ipanepen/ipanepen-add-random-seed
 * adds np.random.seed(0) to test_preprocessing.py to ensure deterministic behavior
 * Initial commit
