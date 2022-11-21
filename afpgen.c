//afpgen.c
//Kevin Kim (kekim@hmc.edu)
#define MASK_EXP 0x7F800000
#define MASK_MANTISSA 0x007FFFFF
#define MASK_SIGN 0x80000000
#define noop

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
//union to access bit-level representation of float32
typedef union {
    float f;
    __uint32_t i;
} U1;


/*
prints msg, and the hex representation and floating point representation of f
NOTE: output can depend on machine endian-ness
*/
void printF32(char * msg, float f)
{
    U1 conv;
    conv.f = f;
    printf("%s: ", msg);
    printf("0x%04x_%04x=%g\n", (conv.i >> 16), (conv.i & 0xFFFF), conv.f);
}

//generates AFP for 16-element blocks
char* genAFP_b16(float * v_in, __uint32_t size_in)
{

    __uint32_t size_out = size_in;
    //__uint32_t num_blocks = size_in/blockSize;

    

}

/*
genAFPHelper_b16

helper function to generate 16-element blocked AFP

v_in: 16-element array of float32
*/
char* genAFPHelper_b16(float * v_in){
    
    //FIND MAX EXPONENT
    unsigned char maxExp = 0;
    bool allPos = 1;

    U1 conv;
    for (short i = 0; i < 16; i++)
    {
        //use union to convert between floating point and bitwise views
        conv.f = v_in[i];

        //logic to determine if all elements are positive
        allPos = allPos && !(conv.i>>31);

        //logic to obtain maxExp
        unsigned char curr = (conv.i & MASK_EXP) >> 23;
        if (curr > maxExp)
        {
            maxExp = curr;
        }
    }
    noop;
    noop;

    


}





int main(){

    U1 conv;
    float f = -16;
    float arr[] = {1028, 256, 8, .0625};
    float out[4];

    genAFPHelper_b16(arr);
    printF32("X", f);
}
