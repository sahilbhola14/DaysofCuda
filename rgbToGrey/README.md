## RGB Image to GreyScale

Summary:
- Vanilla: Convert a RGB Image to greyscale, each thread computes one pixel of the output

Learnings:
- Map from `m x n` output matrix to `m x (3n)' matrix. Since each output pixel utilizes 3 channel (RGB) information from the input image.
- We can obtain the start pointer in the input image and then step through the channels
