## Vector Addition Kernel

Learnings
- Maximum threads per block (so far) is 1024
- Threads block can be sheduled arbitarily, we must not make any assumption
- Global index defined in the kernel is based on the scope of the thread, that is, how
  much data is being processes and how does it access the data
- Functions that are used by both device and host can have `__host__' and `_____device_____' declaration
