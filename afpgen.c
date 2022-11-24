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

purpose: helper function to generate 16-element blocked AFP

v_in: 16-element array of float32

output: pointer to 17 element byte array (16 element of AFP values with shared field is last element)
*/
uint8_t* genAFPHelper_b16(float * v_in){

    //array that will store 16 element AFP values and shared field
    uint8_t result[17];
    
    //FIND MAX EXPONENT
    uint8_t maxExp = 0;
    bool allPos = 1;

    U1 conv;
    for (uint8_t i = 0; i < 16; i++)
    {
        //use union to convert between floating point and bitwise views
        conv.f = v_in[i];

        //logic to determine if all elements are positive
        allPos = allPos && !(conv.i>>31);

        //logic to obtain maxExp
        uint8_t curr = (conv.i & MASK_EXP) >> 23;
        if (curr > maxExp)
        {
            maxExp = curr;
        }
    }

    //assigns 
    for (uint8_t i = 0; i < 16; i++)
    {
        conv.f = v_in[i];
        uint8_t currExp = (conv.i & MASK_EXP) >> 23;
        uint32_t currMantissa = (conv.i & MASK_MANTISSA);
        bool currSign = conv.i & MASK_SIGN;
        


    }

    


}

/**
 * @brief 
 * 
 * @param sign 
 * @param mantissa a 23 bit right-aligned mantissa
 * @param offset a 3 bit offset from the max exponent
 * @return an 8 bit AFP value with [sign, offset, mantissa] with bit width [1,3,4] respectively.
 */
uint8_t roundNearestEven(bool sign, uint32_t mantissa, uint8_t offset) {
    uint32_t result;
    
    //lsb (fourth most significant bit)
    uint32_t lsb = (mantissa >> 19) & 0x1;
    
    //1 bit right of lsb
    uint32_t guardBit = (mantissa >> 18) & 0x1;

    //if any of the bits more than 1 right of the LSB is 1, turn on the sticky
    uint32_t stickyBit = !!(mantissa & 0x3FFF);

    if (sign){
        //NOTE: we need to change the offset if it rounds up from largest value

    }
    else {

    }
}


int main(){

    U1 conv;
    float f = -16;
    float arr[] = {1028, 256, 8, .0625,0,0,0,0,0,0,0,0,0,0,0,0};
    float out[4];

    genAFPHelper_b16(arr);
    printF32("X", f);
}
