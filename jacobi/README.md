## Jacobi for Poisson Equation

Summary:
- Vanilla: Added vanilla jacobi, each thread loads from global memory.

Learnings:
- Global sync between iterations can be done by exiting the kernel, copying the data
