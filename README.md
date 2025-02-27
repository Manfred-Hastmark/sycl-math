# sycl-math
Math library implemented using SYCL

## Building

Make sure to have acpp in path, then run the following commands to build:

### Generic target (recommended)

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release [-DSMATH_BUILD_TESTS=<ON/OFF> -DSMATH_BUILD_SAMPLES=<ON/OFF> -DSMATH_BUILD_OMP=<ON/OFF>] ..
$ make
```
