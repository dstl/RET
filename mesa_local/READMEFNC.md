# Mesa

This package, `mesa`, has been developed to improve performance and add features in keeping with the current code conventions to facilitate incorporating future development from the open source community.

## Release Notes

### Version

This version is 1.0.0

### 30/03/2022 Release Version 1.0.0

This is the first release version.
This repo works with the ret repo, version 1.0.0
This is intended for use by experienced analysts with software and simulation development experience.
This is a deliverable developed under contract PO410000130089.
This is a development framework and therefore up to the developer to ensure validity of results.
The version number listed in the core mesa code base as 0.8.9 reflects the version this was forked from.

## Enhancements

- Extended and improved test coverage
- Fixed reproducibility bug in multigrid
- Added continuous space visualisations as a default option
- Added native background image support for visualisations
- Added custom icon drawing directly from raw svg for visualisations

### Running tests

To run the tests run the following command from your checkout of the `mesa` repository:

```bash
py.test --cov=mesa --cov-report=html
```

## Caveats

This framework is there to make model development easier, as such it is down to the modeller to validate whatever they model.

## Operating platform

This framework should be capable of being run on any platform that can support a python environment and has sufficient computing power.
The only recommended and/or supported platform is Windows 10 at a level consistent with modern business laptops.
