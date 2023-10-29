#include <cstdio>

__device__ void print_status(unsigned tid, unsigned bid){
  printf("Kernel invocation {%d, %d} ", tid, bid);
}