# AFP_rounding

Program converts batch of FP32 numbers into AFP numbers

FP32 numbers to be converted should be put into afp.in, with each value on a newline

Converted AFP values are in afp.out, with each value's line number corresponding with the input value's line number.

NOTE: the converter assumes 16 element blocking. Blocks are partitioned such that the first 16 elements of afp.in is in one block, the next 16 elements are in another, and so on. 

Because we assume 16-element blocking, if afp.in's value count is not divisible by 16 then the last block of afp.out will be 0 padded such that the last block contains 16 elements.
