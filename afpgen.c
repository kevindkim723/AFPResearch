//afpgen.c
//Kevin Kim (kekim@hmc.edu)
#define MASK_EXP 0x7F800000
#define MASK_MANTISSA 0x007FFFFF
#define MASK_SIGN 0x80000000
#define MASK_MANTISSA_AFP 0x780000;
#define noop

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
//union to access bit-level representation of float32
typedef union
{
    float f;
    __uint32_t i;
} U1;

/*
prints msg, and the hex representation and floating point representation of f
NOTE: output can depend on machine endian-ness
*/
void printF32(char *msg, float f)
{
    U1 conv;
    conv.f = f;
    printf("%s: ", msg);
    printf("0x%04x_%04x=%g\n", (conv.i >> 16), (conv.i & 0xFFFF), conv.f);
}

//generates AFP for 16-element blocks
char *genAFP_b16(float *v_in, __uint32_t size_in)
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
uint8_t *genAFPHelper_b16(float *v_in)
{

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
        allPos = allPos && !(conv.i >> 31);

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
 * @param offset a 3 bit right-aligned offset from the max exponent
 * @return an 8 bit AFP value with [sign, offset, mantissa] with bit width [1,3,4] respectively.
 */
uint8_t roundNearestEven(bool signIn, uint32_t mantissaIn, uint8_t offsetIn)
{
    uint8_t result;

    //lsb (fourth most significant bit)
    bool lsb = (mantissaIn >> 19) & 0x1;

    //guardBit: 1 bit right of lsb
    bool guardBit = (mantissaIn >> 18) & 0x1;

    //if any of the bits more than 1 right of the LSB is 1, turn on the sticky
    bool stickyBit = !!(mantissaIn & 0x3FFF);

    //obtain right-aligned 4 bit mantissa (implied leading 1)
    uint8_t mantissa = (mantissaIn >> 19) & 0xF;

    //obtain right-aligned 3 bit offset
    uint8_t offset = offsetIn;

    uint8_t mantissaAddOne = mantissa + 1;
    uint8_t mantissaOut;
    uint8_t offsetOut = offsetIn;

    // ============================================================================
    //  ROUNDING BEGIN
    // ============================================================================
    if (guardBit)
    {
        if (stickyBit)
        {
            //G == 1, S == 1, Round UP
            mantissaOut = mantissaAddOne;
        }
        else
        {
            if (lsb)
            {
                // L = 1, G = 1, S = 0, Round UP
                mantissaOut = mantissaAddOne;
            }
            else
            {
                // L = 0 , G = 1, S = 0, Round DOWN
                mantissaOut = mantissa;
            }
        }
    }
    else
    {
        //G == 0, Round DOWN
        mantissaOut = mantissa;
    }
    // ============================================================================
    //     ROUNDING END
    // ============================================================================

    // ============================================================================
    //     NORMALZE START
    // ============================================================================
    bool overflowMantissa = mantissaOut >> 4;
    bool offsetZero = !(offsetIn);
    bool offsetEight = offsetIn == 8;

    if (overflowMantissa)
    {
        if (offsetZero)
        {
            // Overflow = 1, offset = 0
            // this means that we have offset = 111, mantissa = 1111, round UP scenario
            // don't round up in this case
            mantissaOut = mantissa;
        }
        else if (offsetEight)
        {
            // overflow = 1, offset = 8
            // since we overflow and offset 7 is DENORM, we can't represent the rounded up number
            // don't round up in this case 
            mantissaOut = mantissa;
        }
        else
        {
            // overflow = 1, offset > 0 & offset != 8
            // we must subtract 1 from the offset and set mantissa = 0
            mantissaOut = 0;
            offset = offsetIn-1;
        }
    }
    // handles when offset >= 7

    //offset difference from 7
    uint8_t offsetDiff = offset - 7;

    //mantissa preppended with leading 1
    uint8_t mantissaOutLeadingOne = mantissaOut |= 0x10;
    if (offset == 8 || offset == 9 || offset == 10 || offset == 11)
    {
        //offset LARGE, can still represent mantissa data
        mantissaOut = mantissaOutLeadingOne >> offsetDiff;
        offsetOut = 7;
    }
    else if (offset > 12)
    {
        //offset HUGE, truncate mantissa to 0.
        mantissaOut = 0;
        offsetOut = 7;
    }
    else 
    {
        //offset small (offset <= 7), do NOTHING
        noop;
    }
    // ============================================================================
    //     NORMALIZE END
    // ============================================================================
    

    // pack values into AFP
    result |= signIn << 7;
    result |= offsetOut << 4;
    result |= mantissaOut;
}

int main()
{

    U1 conv;
    float f = -16;
    float arr[] = {1028, 256, 8, .0625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float out[4];

    genAFPHelper_b16(arr);
    printF32("X", f);
}
