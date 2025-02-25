# sycl-math
Math library implemented using SYCL

## Building

Make sure to have acpp in path, then run the following commands to build:

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release [SMATH_BUILD_TESTS=<ON/OFF> SMATH_BUILD_SAMPLES=<ON/OFF> SMATH_BUILD_OMP=<ON/OFF>] ..
$ make
```
### Builds units tests
```bash
SMATH_BUILD_TESTS
```

### Build samples
```bash
SMATH_BUILD_SAMPLES
```

### OMP backend instead of generic
```
SMATH_BUILD_OMP
```
