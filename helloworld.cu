#include <stdio.h>

__global__ void mykernel(void) {
}

int main() {
	mykernel<<<1,1>>>();
	printf("hello, world!\n");
	return 0;
}

