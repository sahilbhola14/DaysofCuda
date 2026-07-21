## NSYS profile
### vecAdd.cu
1. The min time of cudaMalloc, cudaFree is very low, by the median is average is high.
This is becase for the first call there are lazy objects are tare initialized and usually take longer
2. cudaLaunchKernel overhead is very low, it is mostly JIT loading on the fist kernel launch
3. Actual kernel time is very small (2,241 ns). Data size = 0.4 * 1000000 Bytes, that is, 100K float values.
The code is fully bandwidth bound (with nearly zero arithmetic intensity).
\[
\frac{time for compute}{time for memory transfer} = 2.2 / (33 + 18),
\]
that is memory transfer dominates the cost.
To reduce this cost we have the following options.
- pinned memory to speed the copies using cudaMallocHost

Summary: The kernel is memory bound

### vecAdd_pinned.cu
1. For the vecadd kernel, we cannot use


## NCU profile
1.
