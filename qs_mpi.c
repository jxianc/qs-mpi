#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define MAX_FILENAME_LENGTH 256

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
* @brief Write an array of integers to a file.
*
* @param filename The name of the file to write to.
* @param numbers The array of numbers.
* @param nnumbers How many numbers to write.
*/
static void print_numbers(char const * const filename, uint32_t const * const numbers, uint32_t const nnumbers)
{
  FILE * fout;

  /* open file */
  if((fout = fopen(filename, "w")) == NULL) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write the header */
  fprintf(fout, "%d\n", nnumbers);

  /* write numbers to fout */
  for(uint32_t i = 0; i < nnumbers; ++i) {
    fprintf(fout, "%d\n", numbers[i]);
  }

  fclose(fout);
}

/**
* @brief Output the seconds elapsed while sorting. This excludes input and
*        output time. This should be wallclock time, not CPU time.
*
* @param seconds Seconds spent sorting.
*/
static void print_time(double const seconds)
{
  printf("Sort Time: %0.04fs\n", seconds);
}

/**
 * @brief Swap two elements in an array.
 * 
 * @param A The array.
 * @param i The index of the first element.
 * @param j The index of the second element.
*/
void swap(uint32_t* A, int i, int j) 
{
  // printf("swapping %d and %d\n", A[i], A[j]);
  uint32_t temp = A[i];
  A[i] = A[j];
  A[j] = temp;
}

/**
 * @brief Sort an array using serial quick sort.
 * 
 * @param A The array.
 * @param q The start index.
 * @param r The end index.
*/
void serial_qs(uint32_t *A, int q, int r) {
  if (q < r) {
    int x = A[q];
    int s = q;
    for (int i = q + 1; i <= r; i++) {
      if (A[i] <= x) {
        s++;
        swap(A, s, i);
      }
    }
    swap(A, q, s);
    serial_qs(A, q, s - 1);
    serial_qs(A, s + 1, r);
  }
}

/**
 * @brief D
 * 
 * @return The new chunk size of the current processor. 
*/
int parallel_qs(uint32_t** chunk_ptr, int chunk_size, MPI_Comm comm)
{
  uint32_t* chunk = *chunk_ptr;

  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  // base case: when there is only one element
  if (chunk_size == 1) {
    return 1;
  }

  // base case: when there is only one processor
  if (p == 1) {
    serial_qs(chunk, 0, chunk_size - 1);
    return chunk_size;
  }

  // each processor pick a random pivot, gather in root
  uint32_t random_pivot = chunk[rand() % chunk_size];
  uint32_t *random_pivots = NULL;
  if (rank == 0) {
    random_pivots = malloc(p * sizeof(uint32_t));
  }
  MPI_Gather(&random_pivot, 1, MPI_UNSIGNED, random_pivots, 1, MPI_UNSIGNED, 0, comm);

  // root pick the median of random pivots, broadcast to all
  uint32_t pivot;
  if (rank == 0) {
    if (p % 2 == 0) {
      pivot = (random_pivots[p / 2 - 1] + random_pivots[p / 2]) / 2;
    } else {
      pivot = random_pivots[p / 2];
    }
    free(random_pivots);
  }
  MPI_Bcast(&pivot, 1, MPI_UNSIGNED, 0, comm);

  // each processor partition its own chunk using the pivot
  int i = 0;
  int j = chunk_size - 1;
  while (i <= j) {
    while (i < chunk_size && chunk[i] <= pivot) {
      i++;
    }
    while (j >= 0 && chunk[j] > pivot) {
      j--;
    }
    if (i < j) {
      swap(chunk, i, j);
      i++;
      j--;
    }
  }
  int s_size = i;                     // size of the smaller partition
  int l_size = chunk_size - i;        // size of the larger partition

  // gather s_size and l_size in root
  int *s_sizes = NULL;
  int *l_sizes = NULL;
  if (rank == 0) {
    s_sizes = malloc(p * sizeof(int));
    l_sizes = malloc(p * sizeof(int));
  }
  MPI_Gather(&s_size, 1, MPI_INT, s_sizes, 1, MPI_INT, 0, comm);
  MPI_Gather(&l_size, 1, MPI_INT, l_sizes, 1, MPI_INT, 0, comm);

  // find the next root based on s_total and l_total
  int s_total = 0;
  int l_total = 0;
  int next_root = 0;
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      s_total += s_sizes[i];
      l_total += l_sizes[i];
    }

    next_root = ceil((double) (s_total * p / (s_total + l_total)));
    if (next_root == p && l_total > 0) {
      next_root--;
    } else if (next_root == 0 && s_total > 0) {
      next_root++;
    }
  }
  MPI_Bcast(&next_root, 1, MPI_INT, 0, comm);     // broadcast next_root, is needed to determine color for group split

  // calculate chunk size for each processor for each groups, i.e. s_total / s_p or l_total / l_p
  // all to all personalized communication to send and receive partitioned subarrays
  // use a matrix to store the send and receive counts for each processor
  int s_p = 0;
  int l_p = 0;
  int should_receive = 0;
  int* should_receives = NULL;                // number of elements each processor should receive for next round
  int** send_recv_mat = NULL;                 // row vec is recv, col vec is send
  if (rank == 0) {
    s_p = next_root;                          // number of processors in the smaller group
    l_p = p - next_root;                      // number of processors in the larger group
    
    should_receives = malloc(p * sizeof(int));
    int num_per_s = s_p > 0 ? s_total / s_p : 0;
    int num_per_l = l_p > 0 ? l_total / l_p : 0;
    for (int i = 0; i < s_p; i++) {
      should_receives[i] = num_per_s;
    }
    should_receives[s_p - 1] += s_p > 0 ? s_total % s_p : 0;     // add the remain to last p

    for (int i = s_p; i < p; i++) {
      should_receives[i] = num_per_l;
    }
    should_receives[p - 1] += l_p > 0 ? l_total % l_p : 0;      // add the remain to last p

    send_recv_mat = malloc(p * sizeof(int*));
    for (int i = 0; i < p; i++) {
      send_recv_mat[i] = malloc(p * sizeof(int));
      for (int j = 0; j < p; j++) {
        send_recv_mat[i][j] = 0;
      }
    }

    // form the matrix, determine which elements to send to which processor
    int j = 0;
    for (int i = 0; i < next_root; i++) {
      int curr_size = should_receives[i];
      while (curr_size > 0) {
        if (curr_size >= s_sizes[j]) {
          curr_size -= s_sizes[j];
          send_recv_mat[i][j] = s_sizes[j];
          j += 1;
        } else {
          s_sizes[j] -= curr_size;
          send_recv_mat[i][j] = curr_size;
          curr_size = 0;
        }
      }
    }

    j = 0;
    for (int i = next_root; i < p; i++) {
      int curr_size = should_receives[i];
      while (curr_size > 0) {
        if (curr_size >= l_sizes[j]) {
          curr_size -= l_sizes[j];
          send_recv_mat[i][j] = l_sizes[j];
          j += 1;
        } else {
          l_sizes[j] -= curr_size;
          send_recv_mat[i][j] = curr_size;
          curr_size = 0;
        }
      }
    }
  }
  MPI_Scatter(should_receives, 1, MPI_INT, &should_receive, 1, MPI_INT, 0, comm);

  // send the receive counts to all processors
  int recv_count[p];
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      MPI_Send(send_recv_mat[i], p, MPI_INT, i, 0, comm);
    }
  } 
  MPI_Recv(recv_count, p, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

  // send the send counts to all processors with transpose send_recv_mat
  int** send_recv_mat_t = NULL;
  if (rank == 0) {
    send_recv_mat_t = malloc(p * sizeof(int*));
    for (int i = 0; i < p; i++) {
      send_recv_mat_t[i] = malloc(p * sizeof(int));
      for (int j = 0; j < p; j++) {
        send_recv_mat_t[i][j] = send_recv_mat[j][i];
      }
    }
  }
  int send_count[p];
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      MPI_Send(send_recv_mat_t[i], p, MPI_INT, i, 0, comm);
    }
  }
  MPI_Recv(send_count, p, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

  // free allocated memory
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      free(send_recv_mat[i]);
    }
    free(send_recv_mat);

    for (int i = 0; i < p; i++) {
      free(send_recv_mat_t[i]);
    }
    free(send_recv_mat_t);

    free(s_sizes);
    free(l_sizes);
    free(should_receives);
  }

  // calculate send_displs and recv_displs from send_count and recv_count
  int send_displs[p];
  int recv_displs[p];
  int curr_send = 0;
  int curr_recv = 0;
  for (int i = 0; i < p; i++) {
    send_displs[i] = curr_send;
    curr_send += send_count[i];
    recv_displs[i] = curr_recv;
    curr_recv += recv_count[i]; 
  }

  // all to all personalized communication
  uint32_t* new_chunk = malloc(should_receive * sizeof(uint32_t));
  MPI_Alltoallv(chunk, send_count, send_displs, MPI_UNSIGNED, new_chunk, recv_count, recv_displs, MPI_UNSIGNED, comm);

  // split into two groups
  int color = rank < next_root ? 0 : 1;
  MPI_Comm new_comm;
  MPI_Comm_split(comm, color, rank, &new_comm);

  // recursively call parallel_qs on the two groups
  free(*chunk_ptr);
  int new_chunk_size = parallel_qs(&new_chunk, should_receive, new_comm);
  *chunk_ptr = new_chunk;

  return new_chunk_size;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  if (argc != 3) {
    printf("Usage: %s <number of elements> <output file>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int N = atoi(argv[1]);
  char output[MAX_FILENAME_LENGTH];
  strncpy(output, argv[2], MAX_FILENAME_LENGTH);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  // printf("rank %d/%d running on %s\n", rank, p, processor_name);

  // generate random numbers
  int num_per_proc = N / p;
  srand(time(NULL) + rank);
  uint32_t* chunk = malloc(num_per_proc * sizeof(uint32_t));
  for (int i = 0; i < num_per_proc; i++) {
    uint32_t random_num = (uint32_t) rand();
    chunk[i] = random_num;
  }

  // sort
  double start_time = MPI_Wtime();
  int new_chunk_size = parallel_qs(&chunk, num_per_proc, MPI_COMM_WORLD);
  
  // gather all chunks to root
  int* new_chunk_sizes = NULL;
  if (rank == 0) {
    new_chunk_sizes = malloc(p * sizeof(int));
  }
  MPI_Gather(&new_chunk_size, 1, MPI_INT, new_chunk_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // ensure each processor has N / p elements
  int** send_recv_mat = NULL;
  if (rank == 0) {
    int total = 0;
    for (int i = 0; i < p; i++) {
      total += new_chunk_sizes[i];
    }

    int should_have = N / p;
    send_recv_mat = malloc(p * sizeof(int*));
    for (int i = 0; i < p; i++) {
      send_recv_mat[i] = malloc(p * sizeof(int));
      for (int j = 0; j < p; j++) {
        send_recv_mat[i][j] = 0;
      }
    }
    int j = 0;
    for (int i = 0; i < p; i++) {
      int curr_size = should_have;
      while (curr_size > 0) {
        if (curr_size >= new_chunk_sizes[j]) {
          curr_size -= new_chunk_sizes[j];
          send_recv_mat[i][j] = new_chunk_sizes[j];
          j += 1;
        } else {
          new_chunk_sizes[j] -= curr_size;
          send_recv_mat[i][j] = curr_size;
          curr_size = 0;
        }
      }
    }
  }

  int recv_count[p];
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      MPI_Send(send_recv_mat[i], p, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Recv(recv_count, p, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  int** send_recv_mat_t = NULL;
  if (rank == 0) {
    send_recv_mat_t = malloc(p * sizeof(int*));
    for (int i = 0; i < p; i++) {
      send_recv_mat_t[i] = malloc(p * sizeof(int));
      for (int j = 0; j < p; j++) {
        send_recv_mat_t[i][j] = send_recv_mat[j][i];
      }
    }
  }

  int send_count[p];
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      MPI_Send(send_recv_mat_t[i], p, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Recv(send_count, p, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      free(send_recv_mat[i]);
    }
    free(send_recv_mat);

    for (int i = 0; i < p; i++) {
      free(send_recv_mat_t[i]);
    }
    free(send_recv_mat_t);
  }

  // calculate send_displs and recv_displs from send_count and recv_count
  int send_displs[p];
  int recv_displs[p];
  int curr_send = 0;
  int curr_recv = 0;
  for (int i = 0; i < p; i++) {
    send_displs[i] = curr_send;
    curr_send += send_count[i];
    recv_displs[i] = curr_recv;
    curr_recv += recv_count[i]; 
  }

  uint32_t* final_chunk = malloc(N / p * sizeof(uint32_t));
  MPI_Alltoallv(chunk, send_count, send_displs, MPI_UNSIGNED, final_chunk, recv_count, recv_displs, MPI_UNSIGNED, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();
  if (rank == 0) {
    print_time(end_time - start_time);
  }

  // gather all chunks to root
  uint32_t* sorted_chunk = NULL;
  if (rank == 0) {
    sorted_chunk = malloc(N * sizeof(uint32_t));
  }
  MPI_Gather(final_chunk, N / p, MPI_UNSIGNED, sorted_chunk, N / p, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // output sorted_chunk to file
  if (rank == 0) {
    print_numbers(output, sorted_chunk, N);
    free(new_chunk_sizes);
    free(sorted_chunk);
    free(final_chunk);
  }

  // free(chunk);

  MPI_Finalize();
  return 0;
}