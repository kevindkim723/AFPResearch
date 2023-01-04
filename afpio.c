#include "afpgen.h"

void AFPReadWrite(){
    FILE* fp = fopen("afp.in","r");
    int numLines = countLines(fp);
    fclose(fp);
    int linesMod16 = numLines%16;
    int bufSize = linesMod16 == 0 ? numLines : numLines + (16-linesMod16);
    printf("%i\n",bufSize);

}


int countLines(FILE* fp){
    int count = 0;
    char c;
    // Check if file exists
    if (fp == NULL)
    {
        printf("Could not open file\n");
        return 0;
    }
 
    // Extract characters from file and store in character c
    for (c = getc(fp); c != EOF; c = getc(fp))
        if (c == '\n') // Increment count if this character is newline
            count = count + 1;
 
    printf("The file has %d lines\n ", count);
 
    return count;
    //referenced from https://www.geeksforgeeks.org/c-program-count-number-lines-file/
}

int main(){
    AFPReadWrite();
}