# pclines-python

This package implements a PCLines Transform for line detection in images.

Cite:
```
Dubska et al,PCLines - Line detection with parallel coordinates, CVPR 2011
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

```python
A = pcl.accumulate()
```

TBD



# Contribute

If you have a suggestion for improvement, let us know by filling an issue. Or you can fork the project and submit a pull request.

