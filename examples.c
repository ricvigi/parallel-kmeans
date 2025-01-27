do
{
    MPI_Barrier(MPI_COMM_WORLD);
    ...
    MPI_Barrier(MPI_COMM_WORLD);
} while(...);


