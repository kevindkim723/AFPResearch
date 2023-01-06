#Kevin Kim (kekim@hmc.edu)
"""
simple python script that generates N numbers of FP32 values in (LOWER, UPPER) and writes to afp.in
"""
from random import uniform

f = open("afp.in", "w")

#lower bound
LOWER = 0

#upper bound
UPPER = 100

#number of values to generate
N = 100

lines = ""
for i in range(N):
    val = uniform(LOWER,UPPER)
    lines += f"{val}\n"
f.write(lines)
f.close()
