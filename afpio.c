#include "afpgen.c"
void AFPReadWrite(){
    FILE* fp = fopen("afp.in","r");
    int numLines = countLines(fp);
    fclose(fp);
    int linesMod16 = numLines%16;
    int bufSize = linesMod16 == 0 ? numLines : numLines + (16-linesMod16);
    printf("%i\n",bufSize);

    float * buf_FP32 = calloc(bufSize, sizeof(float));


    char *line = NULL;
    char *pEnd;
    size_t nread; 
    size_t len;

    int index=0;

    fp = fopen("afp.in","r");
    printf("HEEEEE\n");
    while ((nread = getline(&line, &len, fp)) != -1){
        buf_FP32[index] = strtof(line, &pEnd);
        printf("%f\n",buf_FP32[index]);
        index++;
    }
    fclose(fp);


    for (int i = 0; i < bufSize/16; i++){
        //input FP32 array
        float FP32_16[16];

        //output AFP array
        uint8_t AFP_16[17];

        //load FP32 vals into 16-element array
        for (int j = 0; j < 16;j++){
           FP32_16[j] = buf_FP32[i*16+j];
        }

        //FP32 to AFP convert
        genAFPHelper_b16(FP32_16, AFP_16);

        //print AFP values
        printAFP(AFP_16);

    }

    


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