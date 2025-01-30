/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
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

/* Function declarations */
void showFileError(int error, char* filename);
int readInput(char* filename, int *lines, int *samples);
int readInput2(char* filename, float* data);
int writeResult(int *classMap, int lines, const char* filename);
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K);

/* Kernel declarations */
__device__ float euclideanDistanceGPU(float* point, float* center);
__global__ void firstStepGPU(float* point, float* center,
							 int* classMap, int* changes);
__global__ void recalculateCentroidsStep1GPU(float* point, int* classMap,
											 int* pointsPerClass, float* auxCentroids);
__global__ void recalculateCentroidsStep2GPU(float* auxCentroids, int* pointsPerClass, float* centroids, float* maxDist);
__global__ void recalculateCentroidsStep3GPU(float* maxDist, float* centroids, float* auxCentroids);

/* GPU constant memory variables */
__constant__ int samples_d;
__constant__ int K_d;
__constant__ int lines_d;

int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	double start, end;
	start = omp_get_wtime();
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

	// Initial centroids
	srand(0);
	int i;
	for(i=0; i<K; i++)
		centroidPos[i]=rand()%lines;

	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);

	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[1024];

	int j;
	int _class;
	float dist, minDist;
	int it=0;
	int changes = 0;
	float maxDist;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}


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

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	int carveout = 100;
	int size = lines*samples*sizeof(float);
	int csize = K*samples*sizeof(float);
	int cmapsize = lines*sizeof(int);
	int ppcsize = K*sizeof(int);
	float* points_d, *centroids_d, *auxCentroids_d, *maxDist_d;
	int* changes_d, *classMap_d, *pointsPerClass_d;

	/* Device memory allocation */
	CHECK_CUDA_CALL( cudaMalloc((void**)&points_d, size) );
	CHECK_CUDA_CALL( cudaMemcpy(points_d, data, size, cudaMemcpyHostToDevice) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&centroids_d, csize) );
	CHECK_CUDA_CALL( cudaMemcpy(centroids_d, centroids, csize, cudaMemcpyHostToDevice) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&classMap_d, cmapsize) );
	CHECK_CUDA_CALL( cudaMemset(classMap_d, 0, cmapsize) );
	/* pointsPerClass_d, auxCentroids_d, changes_d, maxDist_d are initialized in the loop with cudaMemset */
	CHECK_CUDA_CALL( cudaMalloc((void**)&pointsPerClass_d, ppcsize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&auxCentroids_d, csize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&changes_d, sizeof(int)) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&maxDist_d, sizeof(float)) );
	/* Constant memory allocation */
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(samples_d, &samples, sizeof(int)) );
    CHECK_CUDA_CALL( cudaMemcpyToSymbol(K_d, &K, sizeof(int)) );
	CHECK_CUDA_CALL( cudaMemcpyToSymbol(lines_d, &lines, sizeof(int)) );

	/* Set to 100% the amount of shared memory needed */
	CHECK_CUDA_CALL( cudaFuncSetAttribute(firstStepGPU, cudaFuncAttributePreferredSharedMemoryCarveout, carveout) );
	CHECK_CUDA_CALL( cudaFuncSetAttribute(recalculateCentroidsStep2GPU, cudaFuncAttributePreferredSharedMemoryCarveout, carveout) );

	/*
	 *
	 * END CUDA MEMORY ALLOCATION
	 *
	 */
	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	cudaDeviceProp prop;
	CHECK_CUDA_CALL( cudaGetDevice(0) );
	CHECK_CUDA_CALL( cudaGetDeviceProperties(&prop, 0) );
	printf("Max shared memory: %lu\n", prop.sharedMemPerBlock);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "true" : "false");
	printf("32 bit registers per SM: %d\n", prop.regsPerMultiprocessor);
	printf("Shared memory per SM: %lu\n", prop.sharedMemPerMultiprocessor);
	printf("csize: %d\n", csize);

	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	/* We are using a 1 dimensional block and grid size, to maximize occupancy */
	dim3 dimBlock(BLOCK_DIMX);
	dim3 dimGrid(72);
	do
	{
		it++;
		/*
		 *  changes_d: 		  Total number of points that change centroid in this iteration.
		 *  maxDist_d: 		  Will contain the maximum distance between the old and the new centroids.
		 * 					  If it's smaller than maxThreshold, the algorithm terminates
		 *  pointsPerClass_d: Stores at each iteration the number of points that belong to centroid k
		 *  auxCentroids_d:   This array will store the new centroids at each iteration
		 */
		CHECK_CUDA_CALL( cudaMemset(changes_d, 0, sizeof(int)) );
		CHECK_CUDA_CALL( cudaMemset(maxDist_d, FLT_MIN, sizeof(float)) );
		CHECK_CUDA_CALL( cudaMemset(pointsPerClass_d, 0, ppcsize) );
		CHECK_CUDA_CALL( cudaMemset(auxCentroids_d, 0.0, csize) );

		/* First kernel */
		firstStepGPU<<<dimGrid, dimBlock, csize>>>(points_d, centroids_d, classMap_d, changes_d);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Second kernel */
		recalculateCentroidsStep1GPU<<<dimGrid, dimBlock>>>(points_d, classMap_d, pointsPerClass_d, auxCentroids_d);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Third kernel */
		recalculateCentroidsStep2GPU<<<dimGrid, dimBlock, csize>>>(auxCentroids_d, pointsPerClass_d, centroids_d, maxDist_d);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Fourth kernel */
		recalculateCentroidsStep3GPU<<<dimGrid, dimBlock>>>(maxDist_d, centroids_d, auxCentroids_d);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Copy back maxDist_d, auxCentroids_d and classMap_d in host memory */
		CHECK_CUDA_CALL( cudaMemcpy(&maxDist, maxDist_d, sizeof(float), cudaMemcpyDeviceToHost) );

		/* Save the number of changes for this iteration in host memory */
		CHECK_CUDA_CALL( cudaMemcpy(&changes, changes_d, sizeof(int), cudaMemcpyDeviceToHost) );

		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);

	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	CHECK_CUDA_CALL( cudaMemcpy(centroids, auxCentroids_d, csize, cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaMemcpy(classMap, classMap_d, cmapsize, cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	if (changes <= minChanges)
	{
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations)
	{
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else
	{
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
	CHECK_CUDA_CALL( cudaFree(auxCentroids_d) );
	CHECK_CUDA_CALL( cudaFree(maxDist_d) );
	CHECK_CUDA_CALL( cudaFree(changes_d) );
	CHECK_CUDA_CALL( cudaFree(classMap_d) );
	CHECK_CUDA_CALL( cudaFree(pointsPerClass_d) );
	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return EXIT_SUCCESS;
}



/*
 * This kernel implements the first part of the kmeans algorithm logic. It computes the distance of
 * each point from all of the centroids, and saves in classMap[p] the closest centroid to point p.
 * It does so by assigning a thread to each point, and iterating over each centroid.
 */
__global__
void firstStepGPU(float* point   /* in */, 	   /* WHOLE ARRAY */
				  float* center  /* in */,     /* Centroids array */
				  int* classMap  /* in/out */, /* point i belongs to class k */
				  int* changes	 /* out */)	   /* number of changes made */
{
	/* Global thread id */
	int id = threadIdx.x + blockIdx.x * blockDim.x + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	/* id of a thread inside a block */
	int blockId = threadIdx.x;

	extern __shared__ float centers[];
	float minDist = FLT_MAX;
	int _class = 1;
	float res;

	/* Shared memory allocation */
	if ((K_d * samples_d) > BLOCK_DIMX)
	{
		centers[blockId] = center[blockId];
		if (blockId == 0)
		{
			for (int i = 64; i < (K_d * samples_d); i++)
			{
				centers[i] = center[i];
			}
		}
	} else
	{
		centers[blockId] = center[blockId];
	}
	/* Now all the centroids are stored in shared memory */

	__syncthreads();	/* !!!!!!! ATTENTION: Do NOT remove this barrier !!!!!!! */

	if (id < lines_d)
	{
		/* Iterate over each centroid and compute the distance of point id from centroid k */
		for (int k = 0; k < K_d; k++)
		{
			res = euclideanDistanceGPU(&point[id*samples_d], &centers[k*samples_d]);

			/* update classMap[id] if res is smaller than current shortest distance */
			if (res < minDist)
			{
				minDist = res;
				_class = k + 1;
			}
		}
		if (classMap[id] != _class)
		{
			/* variable changes must be incremented atomically */
			atomicAdd(changes, 1);
		}
		/* NOTE: Always update this variable */
		classMap[id] = _class;
	}
}

/*
 * This kernel computes the values needed for the calculation of the new centroids. After the
 * execution, auxCentroids will have stored the (not normalized) sum of all points that belong
 * to each class, and pointsPerClass will contain the count of all points belonging to each class.
 */
__global__
void recalculateCentroidsStep1GPU(float* points,	   /* in */	 /* array of points */
								  int* classMap, 	   /* in */	 /* point i belongs to class k */
								  int* pointsPerClass, /* out */ /* num of points for each class k */
								  float* auxCentroids) /* out */ /* new centroids are computed here */

{
	/* Global thread id */
	int id = threadIdx.x + blockIdx.x * blockDim.x + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	int _class;
	if (id < lines_d)
	{
		_class = classMap[id];

		/* pointsPerClass has to be incremented atomically to avoid race condition over different threads
		 * belonging to the same class */
		atomicAdd(&pointsPerClass[_class - 1], 1);
		for (int j = 0; j < samples_d; j++)
		{
			atomicAdd(&auxCentroids[(_class - 1)*samples_d + j], points[id*samples_d+j]);
		}
	}
}

/*
 * This kernel divides the sum of the values of all the points belonging to class k by the number
 * of points belonging to class k.
 */
__global__
void recalculateCentroidsStep2GPU(float* auxCentroids, /* out */     /* new centroids are computed here */
								  int* pointsPerClass, /* in */      /* num of points for each class k */
								  float* centroids,    /* in */      /* old centroids */
								  float* maxDist)      /* in/out */  /* maxDistance between old and new centroids */
{
	/* Global thread id */
	int id = threadIdx.x + blockIdx.x * blockDim.x + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	/* id of a thread inside a block */
	int blockId = threadIdx.x;
	int _class;
	extern __shared__ int sharedPointsPerClass[];

	/* Copy pointsPerClass into shared memory */
	if (K_d <= BLOCK_DIMX)
	{
		if (blockId < K_d)
		{
			sharedPointsPerClass[blockId] = pointsPerClass[blockId];
		}
	} else
	{
		sharedPointsPerClass[blockId] = pointsPerClass[blockId];
		if (blockId == 0)
		{
			for (int i = 64; i < K_d; i++)
			{
				sharedPointsPerClass[i] = pointsPerClass[i];
			}
		}
	}
	__syncthreads(); /* !!!!!!! ATTENTION: Do NOT remove this barrier !!!!!!! */

	if (id < K_d)
	{
		_class = sharedPointsPerClass[id];
		for (int j = 0; j < samples_d; j++)
		{
			auxCentroids[id*samples_d + j] /= _class;
		}
	}
}


/*
 * This kernel checks whether the distance between the old centroids and the new ones is small
 * enough for us to end the algorithm
 */
__global__
void recalculateCentroidsStep3GPU(float* maxDist, 		/* out */  /* precision in centroid distance */
								  float* centroids,		/* in */   /* old centroids array */
								  float* auxCentroids)	/* in */   /* new centroids */
{
	/* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	float dist;

	if (id < K_d)
	{
		dist = euclideanDistanceGPU(&centroids[id*samples_d], &auxCentroids[id*samples_d]);

		if (dist > *maxDist)
		{
			*maxDist = dist;
		}
	}
	/* Copy the new centroids into the centroids array */
	if (id < K_d * samples_d)
	{
		centroids[id] = auxCentroids[id];
	}
}


/* Computes the euclidean distance between point1 and point2 (both of dimensionality samples_d) */
__device__
float euclideanDistanceGPU(float* point1 /* in  */,   /* Point 1 */
						   float* point2 /* in  */)   /* Point 2 */
{
	float dist = 0.0;
	for (int i = 0; i < samples_d; i++)
	{
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
	}
	dist = sqrtf(dist);
	return (dist);
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
