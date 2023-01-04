#ifndef AFPGEN
#define AFPGEN

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

void printF32(char *msg, float f);
char *genAFP_b16(float *v_in, __uint32_t size_in);
void genAFPHelper_b16(float *v_in, uint8_t * result);
uint8_t roundNearestEven(bool signIn, uint32_t mantissaIn, uint8_t offsetIn);
void printAFP(uint8_t afp_block[]);
uint8_t findLeadingOneOffset(uint8_t mantissa);
int countLines(FILE* fp);
void AFPReadWrite();

#endif



