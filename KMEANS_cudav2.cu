#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <omp.h>


#define MAXLINE 2000
#define MAXCAD 200

#define BLOCK_DIMX 64
#define BLOCK_DIMY 1
#define BLOCK_DIMZ 1
#define GRID_DIMX 72
#define GRID_DIMY 0
#define GRID_DIMZ 0

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

/* Kernel declarations */
__device__ float euclideanDistanceGPU(float* point, float* center, int samples);
__global__ void kmeans_loop(int* changes, float* points,
							float* centroids, float* auxCentroids, float* maxDist,
							int* classMap, int* pointsPerClass);

/* Function declarations */
void showFileError(int error, char* filename);
int readInput(char* filename, int *lines, int *samples);
int readInput2(char* filename, float* data);
int writeResult(int *classMap, int lines, const char* filename);
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K);

/* GPU constant memory variables */
__constant__ int size_d;
__constant__ int csize_d;
__constant__ int cmapsize_d;
__constant__ int ppcsize_d;
__constant__ int minChanges_d;
__constant__ int maxIterations_d;
__constant__ float maxThreshold_d;
__constant__ int samples_d;
__constant__ int K_d;
__constant__ int lines_d;

int main(int argc, char* argv[])
{
    //START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, samples= 0;

	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	printf("number of samples: %d, dimensionality: %d\n", lines, samples);
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K=atoi(argv[2]);
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++)
		centroidPos[i]=rand()%lines;

	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	/*
	 *
	 * START CUDA MEMORY ALLOCATION
	 *
	 */
	int carveout = 100;
	int size = lines*samples*sizeof(float);
	int csize = K*samples*sizeof(float);
	int cmapsize = lines*sizeof(int);
	int ppcsize = K*sizeof(int);

	const int gridsize = (int) ceil((float)lines/BLOCK_DIMX);
	float* points_d, *centroids_d, *auxCentroids_d, *distCentroids_d, *maxDist_d;
	int* changes_d, *classMap_d, *pointsPerClass_d;
    /* Device memory allocation */
	CHECK_CUDA_CALL( cudaMalloc((void**)&points_d, size) );
	CHECK_CUDA_CALL( cudaMemcpy(points_d, data, size, cudaMemcpyHostToDevice) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&centroids_d, csize) );
	CHECK_CUDA_CALL( cudaMemcpy(centroids_d, centroids, csize, cudaMemcpyHostToDevice) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&classMap_d, cmapsize) );
	CHECK_CUDA_CALL( cudaMemset(classMap_d, 0, cmapsize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&distCentroids_d, ppcsize) );
	CHECK_CUDA_CALL( cudaMemset(distCentroids_d, 0.0, ppcsize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&auxCentroids_d, csize) );
	CHECK_CUDA_CALL( cudaMemset(auxCentroids_d, 0.0, csize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&pointsPerClass_d, ppcsize) );
	CHECK_CUDA_CALL( cudaMemset(pointsPerClass_d, 0, ppcsize) );
    /* Constant memory allocation */
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(size_d, &size, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(csize_d, &csize, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(cmapsize_d, &cmapsize, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(ppcsize_d, &ppcsize, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(minChanges_d, &minChanges, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(maxIterations_d, &maxIterations, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(maxThreshold_d, &maxThreshold, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(samples_d, &samples, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(K_d, &K, sizeof(int), cudaMemcpyHostToDevice) );
	CHECK_CUDA_CALL( cudaMemcpyToSymbol(lines_d, &lines, sizeof(int), cudaMemcpyHostToDevice) );
    /* pointsPerClass_d, auxCentroids_d, changes_d, maxDist_d are initialized in the loop with cudaMemset */
	CHECK_CUDA_CALL( cudaMalloc((void**)&pointsPerClass_d, ppcsize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&auxCentroids_d, csize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&changes_d, sizeof(int)) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&maxDist_d, sizeof(float)) );

	/*
	 *
	 * END CUDA MEMORY ALLOCATION
	 *
	 */

	//END CLOCK*****************************************
	end = clock();
	printf("\nMemory allocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int _class;
	float dist, minDist;
	int it= 0;
	int changes = 0;
	float maxDist;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	/* The mean of a point in n dimensions is itself of dimensionality n! */
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	/* ATTENTION: for now, dimBlock must be <= to K*samples */
	dim3 dimBlock(BLOCK_DIMX);
	dim3 dimGrid(gridsize);
	cudaFuncSetAttribute(kmeans_loop, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
	CHECK_CUDA_CALL( cudaMemcpy(centroids_d, centroids, csize, cudaMemcpyHostToDevice) );
	do
	{
	it++;
	CHECK_CUDA_CALL( cudaMemset(maxDist_d, FLT_MIN, sizeof(float)) );
	CHECK_CUDA_CALL( cudaMemset(changes_d, 0, sizeof(int)) );


	kmeans_loop<<<dimBlock, dimGrid, (2*csize) >>>(changes_d,
												   points_d,
												   centroids_d,
												   auxCentroids_d,
												   maxDist_d,
												   classMap_d,
												   pointsPerClass_d);
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	CHECK_CUDA_CALL( cudaMemcpy(&changes, changes_d, sizeof(int), cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaMemcpy(&maxDist, maxDist_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaMemcpy(centroids, auxCentroids_d, csize, cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaMemcpy(classMap, classMap_d, cmapsize, cudaMemcpyDeviceToHost) );

	} while ((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));


	sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
	outputMsg = strcat(outputMsg,line);



/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);

	//END CLOCK*****************************************
	end = clock();
	printf("\nComputation: %f seconds", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = clock();
	//**************************************************



	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);
    CHECK_CUDA_CALL( cudaFree(points_d) );
    CHECK_CUDA_CALL( cudaFree(centroids_d) );
    CHECK_CUDA_CALL( cudaFree(classMap_d) );
    CHECK_CUDA_CALL( cudaFree(distCentroids_d) );
    CHECK_CUDA_CALL( cudaFree(pointsPerClass_d) );
    CHECK_CUDA_CALL( cudaFree(auxCentroids_d) );
    CHECK_CUDA_CALL( cudaFree(changes_d) );
    CHECK_CUDA_CALL( cudaFree(maxDist_d) );

	//END CLOCK*****************************************
	end = clock();
	printf("\n\nMemory deallocation: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	fflush(stdout);
	//***************************************************/
	return 0;
}
/* Computes the euclidean distance between point1 and point2 (both of dimensionality samples) */
__device__
float euclideanDistanceGPU(float* point1  /* in  */,   /* Point 1 */
						   float* point2  /* in  */,   /* Point 2 */
						   int samples	  /* in  */)   /* Dimensionality of points */
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++)
	{
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
	}
	return sqrtf(dist);
}

__global__
void kmeans_loop(int* changes,
                 float* points,
                 float* centroids,
				 float* auxCentroids,
				 float* maxDist,
                 int* classMap,
                 int* pointsPerClass)
{
    /* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	/* id of a thread inside a block */
	int blockId = threadIdx.x;
    int local_changes, _class, j;
    float minDist, dist;

    extern __shared__ float sMem[];
    float* sCentroids = sMem;
    float* sAuxCentroids = (float*) &sMem[csize_d];

    /* store centroids and auxCentroids shared memory */
    if (csize_d <= 64)
    {
        sCentroids[blockId] = centroids[blockId];
        sAuxCentroids[blockId] = 0.0;
    } else
    {
        sCentroids[blockId] = centroids[blockId];
        sAuxCentroids[blockId] = 0.0;
        if (blockId == 0)
        {
            for (int i = 64; i < csize_d; i++)
            {
                sCentroids[i] = centroids[i];
                sAuxCentroids[i] = 0.0;
            }
        }
    }

    __syncthreads(); /* !!!! ATTENTION: do NOT remove this barrier !!!! */
	if (id == 0)
	{
		*changes = 0;
	}
	__syncthreads();
	if (id < lines_d)
	{
		local_changes = 0;
		_class = 1;
		minDist = FLT_MAX;
		/* Compute the distance of each point from each centroid and assign to point id the closest centroid */
		/* ATTENTION: a bug is lurking here */
		for (int k = 0; k < K_d; k++)
		{
			dist = euclideanDistanceGPU(&points[id*samples_d], &sCentroids[k*samples_d], samples_d);
			if (dist < minDist)
			{
				_class = k + 1;
				minDist = dist;
			}
			if (classMap[id] != _class)
			{
				local_changes++;
				classMap[id] = _class;
			}
			// classMap[id] = _class;
		}
		atomicAdd(changes, local_changes);
	}

	__syncthreads();
	if (id < K_d)
	{
		/* NOTE: can we improve this by storing in shared memory a partial sum for each SM? */
		pointsPerClass[id] = 0;
	}
	if (id < K_d*samples_d)
	{
		auxCentroids[id] = 0.0;
		sAuxCentroids[blockId] = 0.0;
	}

	__syncthreads();
	if (id < lines_d)
	{
		_class = classMap[id];
		atomicAdd(&pointsPerClass[_class - 1], 1);
		for (j = 0; j < samples_d; j++)
		{
			atomicAdd(&sAuxCentroids[(_class-1)*samples_d + j], points[id*samples_d + j]);
		}
	}
	__syncthreads();

	if (id < K_d*samples_d)
	{
		int k = id / K_d;
		sAuxCentroids[blockId] /= pointsPerClass[k];
	}
	__syncthreads();
	/* Add to the global array containing the new centroids the local value in shared memory */
	if (id < K_d)
	{
		for(j = 0; j < samples_d; j++)
		{
			atomicAdd(&auxCentroids[id*samples_d + j], sAuxCentroids[blockId]);
		}
	}
	__syncthreads();
	if (id < K_d)
	{
		dist = euclideanDistanceGPU(&centroids[id*samples_d],
									&auxCentroids[id*samples_d],
									samples_d);
		/* BUG: Race condition here in the update of maxDist... how to fix this? */
		if (dist > *maxDist)
		{
			*maxDist = dist;
		}
	}
}

/*
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);
}

/*
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;

    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
        return 0;
    }
    else
	{
    	return -2;
	}
}

/*
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/*
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{
    FILE *fp;

    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);

        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*
Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}


