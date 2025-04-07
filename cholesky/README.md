## Cholesky Factorization

Summary:
- Vanilla: Sequential Cholesky Decomposition (Host)
- Right Looking: Recursive Cholesky Decompostion

Learnings:
- Mapping subsequent smaller matrices to smaller thread blocks.
- Cholesky has three steps: (a) Sqrt of the Diagonal, (b) Normalize all element below the diagonal, and (c) Normalize the lower right block.
- Cholesky can be computed recursively using the right looking algorithm: https://www.cs.utexas.edu/~flame/laff/alaff/chapter05-cholesky-right-looking-algorithm.html
