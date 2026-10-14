## changelog

### unreleased

nothing yet!

### [0.1.0] (2026-10-14)

Initial version of the plugin.
Most of the code are based on `nao-manga-rls` stuff by me.

Implemented:
- `Levels` function, to adjust the brightness and contrast of the image.
- `Posterize` function, to reduce the number of colors in the image.
- `Deblur` function, to reduce the blur in the image.
  - Method 0 is based on Richardson-Lucy deconvolution algorithm.
  - Method 1 is unsharp mask algorithm.
- `PeakStats` function, to analyze the image and return the peak values of luma channel.
- `PeakGrayShades` function, to analyze the image and return the amount of gray shades in the image.
