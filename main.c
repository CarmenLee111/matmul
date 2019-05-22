#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

int check_sqrt(int n);
int load_input(int** A, int** B, char *filename);
void write_output(int n, int* array, char* filename);
void print_matrix(int* A, int n);
void mmultiply(int* a, int *b, int *c, int n);

int main(int argc, char *argv[]) {



  /* Retrieve the arguments */
  char *inputfile = argv[1];
  char *outputfile = argv[2];

  /* global variables */
  int size, rank, rank_cart;
  int i, j, k, n;               /* n - size of the matrix */
  int s;                        /* number of shifts */
  int chunk;                    /* Amount of work each processor will do */
  int *A, *B, *C;               /* matrix A, B, and C */
  int *a, *b, *c;               /* submatrix A, B, C and tmp */
  int d[2];                     /* dimensions of the processors */    
  int periods[2];               /* determine whether the topology is periodic */
  int coord[2];                 /* coordinates of the cartesian topology */ 
  int left, right, up, down;    /* the adjacent processors of the focal one */
  int alignsrc, aligndes;       /* source and destination for alignment shift */
  double starttime, t;

  MPI_Comm  comm_cart;
  MPI_Request request;
  MPI_Status  status;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);       

  /* Global IDs */
  MPI_Comm_size(MPI_COMM_WORLD, &size);     
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);    


  d[0] = (int) sqrt((float)size); d[1] = (int) sqrt((float) size);
  periods[0] = 1; periods[1] = 1;
  
   /* Dimensions of the processors */
  //MPI_Dims_create(size, 2, d);    /* did not work with n=9...*/   
  //printf("Dimension of PEs: (%d, %d)\n", d[0], d[1]);                                   
  
  if (rank == 0) {
    /* Usage */
    if (argc != 3) {
        printf("Usage: ./matmul <inputfilename> <outputfilename>\n");
        return -1;
    } 

    /* Check if the processors are a square of an integer */ 
    if (sqrt((float)size) != (int) sqrt((float) size)){
        printf("Square root of p must be an integer\n");
        return -1;
    }

    /* Loading the input file into l */
    n = load_input(&A, &B, inputfile);      /* return size of the square matrix*/

    /* Quit program if condition not met */
    if (n%d[0]!=0) {
        printf("n/sqrt(p) must be an integer!\n");
        return -1;
    }

    /* chunk size of the matrix in each process */
    chunk = n/d[0];

    // print_matrix(A, n);
    // print_matrix(B, n);

    // printf("Dimensions of the processors: %d, %d\n", d[0], d[1]);
  }
  
  
  starttime = MPI_Wtime();  

  C = (int *) malloc(n*n*sizeof(int));

  /* -------------------- data to local matrix  --------------- */
  MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Datatype matrix;
  MPI_Datatype array;

  MPI_Type_vector(chunk, chunk, n, MPI_INT, &array);
  MPI_Type_create_resized(array, 0, sizeof(int), &matrix);
  MPI_Type_commit(&matrix);

  /* Local sub matrix memloc */
  a   = (int*) malloc(chunk * chunk * sizeof(int));
  b   = (int*) malloc(chunk * chunk * sizeof(int));
  c   = (int*) calloc(chunk * chunk, sizeof(int));

  int counts[d[0] * d[1]];
  int displs[d[0] * d[1]];

  for (i=0; i<d[0]; i++) {
      for (j=0; j<d[1]; j++) {
          counts[i*d[1]+j] = 1;
          displs[i*d[1]+j] = i*n*chunk + j*chunk;
      }
  }
//   for (i=0; i<d[0]*d[1]; i++){
//       printf("%d ", counts[i]);
//   }
//   printf("\n");

//   for (i=0; i<d[0]*d[1]; i++){
//       printf("%d ", displs[i]);
//   }
//   printf("\n");

  /* Scatter A and B to the processors */
  MPI_Scatterv(A, counts, displs, matrix, a, chunk*chunk, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(B, counts, displs, matrix, b, chunk*chunk, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  
  /* print and inspect */
  // printf("Rank %d has: \n", rank);
  // print_matrix(a, chunk);
  // print_matrix(b, chunk);

  /* ------------ Processor topology and alignment -------------- */    
  /* Create the Cartesian communicators */
  MPI_Cart_create(MPI_COMM_WORLD, 2, d, periods, 1, &comm_cart); 

  /* Get the Cartesian rank and coordinates */
  MPI_Comm_rank(comm_cart, &rank_cart);
  MPI_Cart_coords(comm_cart, rank_cart, 2, coord);
  // printf("Rank %d has Cart rank %d and coordinates; (%d, %d)\n", rank, rank_cart, coord[0], coord[1]);

  /* Get local ranks of the adjacent PEs for the shifting */
  MPI_Cart_shift(comm_cart, 1, -1, &right, &left);   /* dir=1, j varies, row shift, E or W */
  MPI_Cart_shift(comm_cart, 0, -1, &down,  &up);
  // printf("Rank %d has horizontal neighbors: (%d, %d), vertical nbs: (%d, %d)\n", rank, left, right, up, down);

  
  /* Alignment */
  MPI_Cart_shift(comm_cart, 1, -coord[0], &alignsrc, &aligndes);  /* E or W based on i */
  MPI_Sendrecv_replace(a, chunk*chunk, MPI_INT, aligndes, 11, alignsrc, 11, comm_cart, &status);

  MPI_Cart_shift(comm_cart, 0, -coord[1], &alignsrc, &aligndes);  /* N or S based on j */
  MPI_Sendrecv_replace(b, chunk*chunk, MPI_INT, aligndes, 11, alignsrc, 11, comm_cart, &status);

  /* print and inspect */
  // printf("Rank %d has a and b: \n", rank);
  // print_matrix(a, chunk);
  // print_matrix(b, chunk);
  MPI_Barrier(MPI_COMM_WORLD); 

  mmultiply(a, b, c, chunk);
  // printf("Rank %d has c: \n", rank); 
  // print_matrix(c, chunk);

  /* shift and compute */
  for (i=0; i<d[0]-1; i++) {
      MPI_Sendrecv_replace(a, chunk*chunk, MPI_INT, left, 11, right, 11, comm_cart, &status);
      MPI_Sendrecv_replace(b, chunk*chunk, MPI_INT, up, 11, down, 11, comm_cart, &status);
      mmultiply(a, b, c, chunk);
    //   printf("Rank %d has c: \n", rank); 
    //   print_matrix(c, chunk);
   }

  /* Gather C from the processors to master */
  MPI_Gatherv(c, chunk*chunk, MPI_INT, C, counts, displs, matrix, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Inspect the result */
  // if (rank==0) {
    //  print_matrix(C, n);
  // }

  t = MPI_Wtime() - starttime;

  if (rank==0) {
     /* WALL TIME */
     printf("%f\n", t);

     /* Write to output (ommitted for measuring time) */ 
     write_output(n, C, outputfile);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Type_free(&matrix);
  free(a);
  free(b);
  free(c);
  if (rank == 0 ) {
    free(A);
    free(B);
  }
  free(C);

  MPI_Finalize();                  /* Shut down and clean up MPI */

  return 0;
}


int load_input(int** A, int** B, char *filename) {
  FILE *fp = fopen(filename, "r");
  int n;
  if (!fp) {
    printf("load_data error: failed to open input file '%s'.\n", filename);
    return -1;
  }
  fscanf(fp, "%d", &n);
  *A = malloc(sizeof(int) * n * n);
  *B = malloc(sizeof(int) * n * n);
  int i;
  for (i=0; i<n*n; i++) {
      fscanf(fp, "%d", &((*A)[i]));
  }
  for (i=0; i<n*n; i++) {
      fscanf(fp, "%d", &((*B)[i]));
  }
  fclose(fp);
  return n;
}

void write_output(int n, int* array, char* filename) {
  FILE *fp = fopen(filename, "w");
  int i;
  for (i=0; i<n*n; i++) {
      fprintf(fp, "%.6f ", (float) array[i]);
      if ((i+1)%n == 0) fprintf(fp, "\n");
  }
  fclose(fp);
}

void print_matrix(int* A, int n){
      int i,j;
    for (i=0; i<n*n; i++) {
        printf("%d ", A[i]);
        if ((i+1)%n == 0) printf("\n");
    }
    printf("\n");
}

void mmultiply(int* a, int *b, int *c, int n) {
    int i, j, k;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            for (k=0; k<n; k++) {
                c[i*n+j] += a[i*n+k] * b[k*n+j];
            }
        }
    }
}
