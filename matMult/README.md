## Matrix Multiplication

Summary:
- Vanilla: Each thread computes one element of the output matrix
- Tiled: Each thread computes one element of the output matrix, however, `{tile \times tile}`
  chunk of data is first transferred from DRAM to __shared.

Learnings:
- Use of shared memory for reducing DRAM traffic. (This increases the computations / bytes of data from DRAM)
- _`__synchthreads` should be used when working with shared memory, as threads must be synched.
- Allocating shared memory in static way.
