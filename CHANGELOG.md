# Changelog

All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.5] - 2022-08-15

### Fixed

- Fixed bug when there are no rows to tweak

## [0.1.4] - 2022-08-14

### Changed

- Refactor

### Fixed

- Fixed bug with visualization, when there are no categorical features

## [0.1.2] - 2022-08-10
 
### Added

- Added changelog
   
### Changed

- Limit of counterfactual examples can be 'None' from now
 
### Fixed

- Fixed bug when there are no counterfactuals
- Calculating loss had bug, because it used wrong normalization
- HOEM metric now works properly, when features are out of feature range
