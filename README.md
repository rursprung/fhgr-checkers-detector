# Checkers Detector

## Installation / Dependencies
Since this uses [OpenCV](https://opencv.org/) using the [`opencv` crate](https://crates.io/crates/opencv) you first
need to install OpenCV in some way. This has been tested by installing OpenCV using [`vcpkg`](https://vcpkg.io/):
```bash
${VCPKG_ROOT}/vcpkg install opencv[ffmpeg,contrib]
```
If you do not wish to use `vcpkg` then you should disable the default features.

This has been tested against OpenCV 4.11 and is not guaranteed to work with other (especially older) versions.
