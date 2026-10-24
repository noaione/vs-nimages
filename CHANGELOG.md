## changelog

### unreleased

**Added**:
- `gpu` argument on every filter, which runs that filter on the Vulkan
  device. `1` needs VapourSynth R80 and a Vulkan 1.4 device, `0` (the default)
  is the host path.
  - `Deblur(gpu=1)` runs both methods over every format the host path takes,
    uploads a clip that is not on the device with `std.GPUUpload`, and hands
    back a device frame.
  - 8 bit deblur is byte-identical to `gpu=0`, 9 to 16 bit within one code
    value, the float formats within two ULP, and a clip whose format varies is
    refused because the plan is built once.
  - `Levels(gpu=1)` and `Posterize(gpu=1)` map the level table on the device
    and are byte-identical to `gpu=0`. Under `gpu=1` a float clip is refused by
    `Levels`, whose float curve has no table, and a clip whose format varies is
    refused by both.
  - `PeakStats(gpu=1)` and `PeakGrayShades(gpu=1)` count the histogram on the
    device and analyse it with the same host code, so the properties, the shade
    arrays and the pixels match `gpu=0`. `Posterize(method=1, gpu=1)` solves
    its levels from that table without the frame crossing the bus.
  - Device deblur is 5.6x to 7.8x the host path at 12 MP with 13 frames in
    flight. The analyzers and the maps pay only inside a device chain, where
    the frame is already there, and a lone analyzer loses to the host because
    it is transfer bound.
- `PeakStats` and `PeakGrayShades` accept a `YUV` or `RGB` clip next to a
  `Gray` one. A `YUV` clip is analysed through plane 0, which is luma, and an
  `RGB` clip through the fixed-point luma of its three planes, the same numbers
  `Image.convert("L")` produces, so one page reports the same result in every
  family.

**Changed**:
- Bump Vapoursynth to R80
- The `RGB` luma is computed through `std::arch` kernels on x86-64 and the 8
  bit counter set is striped, so counting an `RGB24` page costs 1.1 to 1.5
  times the gray pass instead of three times.

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
