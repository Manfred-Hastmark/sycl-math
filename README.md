# sycl-math
Math library implemented using SYCL

## Building

Make sure to have acpp in path, then run the following commands to build:

### Generic target (recommended)

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```

### OMP target

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DTARGET_OMP=ON ..
$ make
```

### With unit-tests

```bash
$ mkdir -v build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON ..
$ make
```
