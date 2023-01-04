# AFP_rounding

Program converts batch of FP32 numbers into AFP numbers

FP32 numbers to be converted should be put into afp.in, with each value on a newline

Converted AFP values are in afp.out, with each value's line number corresponding with the input value's line number.

NOTE: the converter assumes 16 element blocking. Blocks are partitioned such that the first 16 elements of afp.in is in one block, the next 16 elements are in another, and so on.

