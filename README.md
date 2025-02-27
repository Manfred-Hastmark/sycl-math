# sycl-math
Math library implemented using SYCL

## Building

Make sure to have acpp in path, then run the following commands to build:

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release [-DSMATH_BUILD_TESTS=<ON/OFF> -DSMATH_BUILD_SAMPLES=<ON/OFF> -DSMATH_BUILD_OMP=<ON/OFF>] ..
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
```bash
SMATH_BUILD_OMP
```
