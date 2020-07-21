# Changelog

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
