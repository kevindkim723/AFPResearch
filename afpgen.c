//afpgen.c
//Kevin Kim (kekim@hmc.edu)
#define MASK_EXP 0x7F800000
#define MASK_MANTISSA 0x007FFFFF
#define MASK_SIGN 0x80000000
#define MASK_MANTISSA_AFP 0x780000;
#define noop

#include "afpgen.h"
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

/*
accepts msg and a 17-byte array of AFP values where the last element is AFP.
NOTE: output can depend on machine endian-ness
*/
void printAFP(uint8_t afp_block[])
{
    uint8_t maxExp = afp_block[16] & 0xFF;
    float f;
    U1 conv;
    for (int i = 0; i < 15; i++)
    {
        conv.f = f;
    
        uint8_t curr  = afp_block[i];
        uint8_t offset = (curr >> 4) & 0x7;
        uint8_t exp = maxExp - offset;
        uint8_t mantissa = curr & 0xF;
        bool sign = curr >> 7;

        if (offset == 0x7){
            if (mantissa == 0)
            {
                // OFFSET = 3b111, MANTISSA = 4b0 represents 0
                exp = 0;
                mantissa = 0;
            }
            else{
                // OFFSET = 3b111, MANTISSA > 0
                // AFP is denorm, we need to use a leading one detector
                uint8_t leadingOneOffset = findLeadingOneOffset(mantissa);
                exp = exp - leadingOneOffset;
                mantissa = mantissa << leadingOneOffset;
            }
        }
        else{
            //AFP not denorm, proceed normally

        }

        //ERRONEOUS... edit: fixed?
        conv.i |= (exp << 23);
        conv.i |= (mantissa << 19);
        conv.i |= (sign << 31);

        printf("AFP %d: %.6f\n", i, conv.f);
    }
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

result: pointer to 17 element byte array (16 element of AFP values with shared field is last element)
*/
void genAFPHelper_b16(float *v_in, uint8_t* result)
{

    //array that will store 16 element AFP values and shared field
    //uint8_t result[17];

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

    //assigns blocked AFP
    for (uint8_t i = 0; i < 16; i++)
    {
        conv.f = v_in[i];

        //extract sign, offset, and mantissa fields
        uint8_t currExp = (conv.i & MASK_EXP) >> 23;
        uint8_t currOffset = maxExp - currExp;
        uint32_t currMantissa = (conv.i & MASK_MANTISSA);
        bool currSign = conv.i & MASK_SIGN;

        //convert to AFP
        result[i] = roundNearestEven(currSign, currMantissa, currOffset);
    }
    
    //pack shared field
    uint8_t shared;
    shared |= maxExp;
    shared |= (allPos << 7);

    //assign shared field as last element in the block
    result[16] = shared;

}

/**
 * @brief 
 * 
 * @param sign 
 * @param mantissa a 23 bit right-aligned mantissa
 * @param offset a 3 bit right-aligned offset from the max exponent
 * @return an 8 bit AFP value with [sign, offset, mantissa] with bit width [1,3,4] respectively.
 */
//TODO: handle when the input is denorm, NAN, and inf
uint8_t roundNearestEven(bool signIn, uint32_t mantissaIn, uint8_t offsetIn)
{
    uint8_t result = 0;;

    //lsb (fourth most significant bit)
    bool lsb = (mantissaIn >> 19) & 0x1;

    //guardBit: 1 bit right of lsb
    bool guardBit = (mantissaIn >> 18) & 0x1;

    //if any of the bits more than 1 right of the LSB is 1, turn on the sticky
    bool stickyBit = !!(mantissaIn & 0x3FFFF);

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
    bool offsetGTseven = (offset > 7);

    if (overflowMantissa)
    {
        if (offsetZero)
        {
            // Overflow = 1, offset = 0
            // this means that we have exp = 111, mantissa = 1111, round UP scenario
            // don't round up in this case
            mantissaOut = mantissa;
        }
        else
        {
            // overflow = 1, offset > 0
            // we must subtract 1 from the offset and set mantissa = 0
            mantissaOut = 0;
            offset = offsetIn-1;
        }
    }

    // ============================================================================
    //     NORMALIZE END
    // ============================================================================
    // NOTE: at this point we have an offset and a 4-bit mantissa in mantissaOut with an implied leading 1. If offset >= 7 then we must convert to a denorm.
    // mantissaOut = 1.mmmm

    // ============================================================================
    //     HANDLE DENORM BEGIN
    // ============================================================================

    // NOTE: at this stage, our offset could be >= 7 so we swizzle our number if it's a denorm.
    //offset difference from 7
    uint8_t offsetDiff = offset - 7;

    //mantissa preppended with leading 1
    uint8_t mantissaOutLeadingOne = mantissaOut;
    mantissaOutLeadingOne |= 0x10;
    if (offset == 8 || offset == 9 || offset == 10 || offset == 11)
    {
        //offset LARGE, can still represent mantissa data

        //shift accordingly
        mantissaOut = mantissaOutLeadingOne >> (offsetDiff+1);
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
        offsetOut = offset;
        noop;
    }
    // ============================================================================
    //     HANDLE DENORM END
    // ============================================================================


    // pack values into AFP
    result |= (signIn << 7);
    result |= (offsetOut << 4);
    result |= mantissaOut;

    return result;
}

int main()
{

    U1 conv;
    float f = -16;
    float arr[] = {1028, 1024, 300, .0625, 256, 1000, 923.33, 56.2, 0, 0, 0, 0, 0, 0, 0, 0,0};
    float out[4];

    uint8_t result[17];
    genAFPHelper_b16(arr,result);
    printF32("X", f);
    for (int i = 0 ; i< 16 ;i ++)
    {
        printF32("x", arr[i]);
    }
    printAFP(result);


}
/**
 * @brief 
 * returns the offset from MSB of leading one (must be geq 1)
 * this is used when converting a denorm AFP to FP32. There is no implicit one so we must use a leading one detector.
 * 
 * @param mantissa 4 bit AFP mantisssa
 * @return uint8_t offset of the leading one
 */
uint8_t findLeadingOneOffset(uint8_t mantissa){
    if (mantissa>>3) return 1;
    else if (mantissa>>2) return 2;
    else if (mantissa>>1) return 3;
    else return 4; //this shouldn't happen and by extension shouldn't matter.
}