# `pclines` package for Python

![pclines](doc/pclines.svg)

This package implements a PCLines transform for line detection in images.

```bibtex
@INPROCEEDINGS{dubska2011pclines,
    author={M. {Dubská} and A. {Herout} and J. {Havel}},
    booktitle={CVPR 2011},
    title={PClines — Line detection using parallel coordinates},
    year={2011},
}
```

# Requrements

* Python 3.6+
* numpy
* numba
* scikit-image

# Installation

The package is on [PyPI](https://pypi.org/project/pclines/), so just run following command and install the package.

```shell
> pip install pclines
```

Alternatively, you can download this repository and install manually.


# Example

1. Import package

```python
import pclines as pcl
```

2. Accumulation of observations
The observations are 2D weighted coordinates enclosed by a known bounding box.

*TBD*



# Contribute

If you have a suggestion for improvement, let us know by filling an issue. Or you can fork the project and submit a pull request.

