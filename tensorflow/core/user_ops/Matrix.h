#include <stdlib.h>
#include <stdio.h>
#include "riffa.h"
#define ID 0;
#define CHNL 0;
int sendToFpga( int *a, int a_m, int a_n,  int *b, int b_m, int b_n)
{
	fpga_t * fpga;
	int send;
    int id = ID;
    int chnl = CHNL;
    int * sendBuffer;
	int err;
	size_t maxWords;
	int a_num = a_m*a_n;
	int b_num = b_m*b_n;
	maxWords = a_num + b_num;
	printf("send %zu words\n",  maxWords);
	// Get the device with id
	fpga = fpga_open(id);
	if (fpga == NULL) {
		printf("Could not get FPGA %d\n", id);
		return -1;
	}
	// Malloc the arrays
	sendBuffer = (int *)malloc(maxWords*4);
	if (sendBuffer == NULL) {
		printf("Could not malloc memory for sendBuffer\n");
		fpga_close(fpga);
		return -1;
	}
	//conbine two array to one array
    int j,k;
	for(j = 0; j < a_num; j++)
	{
		sendBuffer[j] = a[j];
	}
	for( k = 0; k < b_num; k++)
	{
		sendBuffer[k + a_num] = b[k];
	}
	send = fpga_send(fpga, chnl, sendBuffer, maxWords, 0, 1, 25000);
    for(j =0;j<maxWords;j++)
    {
        printf("sendBuffer result:%d\n",sendBuffer[j]);
    }
	printf("Test %d: words send: %d\n", j, send);
	fpga_close(fpga);
	return send;
}

int recvFromFpga(int **a, int m, int n){
	fpga_t * fpga;
	int id = ID;
    int chnl = CHNL;
    int recvd;
	int * recvBuffer;
	int err;
	size_t maxWords;
	maxWords = m*n;
	// Get the device with id
	fpga = fpga_open(id);
	if (fpga == NULL) {
		printf("Could not get FPGA %d\n", id);
		return -1;
	}
	// Malloc the arrays
	recvBuffer = (int *)malloc(maxWords*4 + 4);
	recvBuffer +=1;
	if (recvBuffer == NULL) {
		printf("Could not malloc memory for recvBuffer\n");
		fpga_close(fpga);
		return -1;
	}
	recvd = fpga_recv(fpga, chnl, recvBuffer, maxWords, 25000);
	int j,k;
    for(j = 0; j < m; j++)
	{
		for( k = 0; k < n; k++)
		{
            printf("recvbuffer result:%d\n",recvBuffer[j * n + k]);
			a[j][k] = recvBuffer[j * n + k];
		}
	}
	fpga_close(fpga);
    return recvd;
}
/*
int main(int argc, char** argv) {
	int *a = (int *)malloc(16);
	int *b = (int *)malloc(16);
    int i;
    for(i = 0; i < 4; i++)
    {
        a[i] = i+1;
        b[i] = i+5;
    }
	sendToFpga(a,2,2,b,2,2);
    printf("send succeed!\n");
	int **c = (int **)malloc(sizeof(int *)*2);
    
    for(i = 0; i < 2; i++)
    {
        c[i] = (int *)malloc(8);
    }
    printf("begin recv!\n");
	recvFromFpga(c,2,2);
    printf("recv end!");
    int j,k;
	for(j = 0; j < 2; j++)
	{
		for(k = 0; k < 2; k++)
		{
			printf("result :%d\n",c[j][k]);
		}
	}
    free(a);
    free(b);
    for(i = 0; i < 2; i++)
    {
        free(c[i]);
    }
    free(c);
}
*/
