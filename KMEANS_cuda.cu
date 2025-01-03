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

/* Function declarations */
void showFileError(int error, char* filename);
int readInput(char* filename, int *lines, int *samples);
int readInput2(char* filename, float* data);
int writeResult(int *classMap, int lines, const char* filename);
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K);

/* Kernel declarations */
__device__ float euclideanDistanceGPU(float* point, float* center, int samples);
__global__ void firstStepGPU(float* point, float* center,
							 int* classMap, int samples,
							 int lines, int K, int* change);
__global__ void recalculateCentroidsStep1GPU(float* point, int* classMap,
											 int* pointsPerClass, float* auxCentroids,
											 int lines, int K, int samples);
__global__ void recalculateCentroidsStep2GPU(float* auxCentroids, int* pointsPerClass,
											 int K, int samples);
__global__ void recalculateCentroidsStep3GPU(float* maxDist, float* centroids, float* auxCentroids,
											 int samples, int K);

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


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

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

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
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
	/* pointsPerClass_d, auxCentroids_d, changes_d, maxDist_d are initialized in the loop with cudaMemset */
	CHECK_CUDA_CALL( cudaMalloc((void**)&pointsPerClass_d, ppcsize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&auxCentroids_d, csize) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&changes_d, sizeof(int)) );
	CHECK_CUDA_CALL( cudaMalloc((void**)&maxDist_d, sizeof(float)) );

	/* ATTENTION: for now, dimBlock must be <= to K*samples */
	dim3 dimBlock(BLOCK_DIMX);
	dim3 dimGrid(gridsize);
	do{
		it++;

		CHECK_CUDA_CALL( cudaMemset(changes_d, 0, sizeof(int)) );
		CHECK_CUDA_CALL( cudaMemcpy(centroids_d, centroids, csize, cudaMemcpyHostToDevice) );

		/* Set to 100% the amount of shared memory needed */
		cudaFuncSetAttribute(firstStepGPU, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
		/* First kernel */
		firstStepGPU<<<dimGrid, dimBlock, csize>>>(points_d,
												   centroids_d,
												   classMap_d,
												   samples,
												   lines,
												   K,
												   changes_d);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Zero out pointsPerClass_d and auxCentroids_d for this iteration */
		CHECK_CUDA_CALL( cudaMemset(pointsPerClass_d, 0, ppcsize) );
		CHECK_CUDA_CALL( cudaMemset(auxCentroids_d, 0.0, csize) );
		/* Second kernel */
		recalculateCentroidsStep1GPU<<<dimGrid, dimBlock>>>(points_d,
															classMap_d,
															pointsPerClass_d,
															auxCentroids_d,
															lines,
															K,
															samples);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Third kernel */
		recalculateCentroidsStep2GPU<<<dimGrid, dimBlock, csize>>>(auxCentroids_d,
																   pointsPerClass_d,
																   K,
																   samples);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		CHECK_CUDA_CALL( cudaMemset(maxDist_d, FLT_MIN, sizeof(float)) );
		/* Fourth kernel */
		recalculateCentroidsStep3GPU<<<dimGrid, dimBlock>>>(maxDist_d,
															centroids_d,
															auxCentroids_d,
															samples,
															K);
		CHECK_CUDA_CALL( cudaDeviceSynchronize() );

		/* Copy back maxDist_d, auxCentroids_d and classMap_d in host memory */
		CHECK_CUDA_CALL( cudaMemcpy(&maxDist, maxDist_d, sizeof(float), cudaMemcpyDeviceToHost) );
		CHECK_CUDA_CALL( cudaMemcpy(centroids, auxCentroids_d, csize, cudaMemcpyDeviceToHost) );
		CHECK_CUDA_CALL( cudaMemcpy(classMap, classMap_d, cmapsize, cudaMemcpyDeviceToHost) );
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

	CHECK_CUDA_CALL( cudaMemcpy(classMap, classMap_d, cmapsize, cudaMemcpyDeviceToHost) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
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
	CHECK_CUDA_CALL( cudaFree(auxCentroids_d) );
	CHECK_CUDA_CALL( cudaFree(distCentroids_d) );
	CHECK_CUDA_CALL( cudaFree(changes_d) );
	CHECK_CUDA_CALL( cudaFree(classMap_d) );
	CHECK_CUDA_CALL( cudaFree(pointsPerClass_d) );
	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}



/* This kernel implements the first part of the kmeans algorithm logic. It computes the distance of
 * each point from all of the centroids, and saves in classMap[p] the closest centroid to point p.
 * It does so by assigning a thread to each point, and iterating over each centroid. NOTE: Can this
 * be optimized further? */
__global__
void firstStepGPU(float* point   /* in */, 	   /* WHOLE ARRAY */
				  float* center  /* in */,     /* Centroids array */
				  int* classMap  /* in/out */, /* point i belongs to class k */
				  int samples    /* in */, 	   /* dimensionality of points */
				  int lines      /* in */, 	   /* number of points */
				  int K			 /* in */, 	   /* number of centroids */
				  int* changes	 /* out */)	   /* number of changes made */
{
	/* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	/* id of a thread inside a block */
	int blockId = threadIdx.x;

	/* ATTENTION: Think wisely about the size of these shared memory arrays. REMEMBER that
	 * we can store at max 16384 floats in shared memory in a 7.5 CC GPU */
	extern __shared__ float centers[];
	float minDist = FLT_MAX;
	int local_changes = 0;
	int _class;
	float res;

	/* Shared memory allocation. NOTE: It's much quicker (and better) if K*samples < 64. This should always
	 * be possible, except in the case where we use 100 dim points */
	if ((K * samples) > BLOCK_DIMX)
	{
		centers[blockId] = center[blockId];
		if (blockId == 0)
		{
			for (int i = 64; i < (K * samples); i++)
			{
				centers[i] = center[i];
			}
		}
	} else
	{
		if (blockId < (K*samples))
		{
			centers[blockId] = center[blockId];
		}
	}
	/* Now all the centroids are stored in shared memory */

	__syncthreads();	/* !!!!!!! ATTENTION: Do NOT remove this barrier !!!!!!! */

	if (id < lines)
	{
		/* Iterate over each centroid and compute the distance of point id from centroid k */
		for (int k = 0; k < K; k++)
		{
			res = euclideanDistanceGPU(&point[id*samples], &centers[k*samples], samples);

			/* update classMap[id] if res is smaller than current shortest distance */
			if (res < minDist)
			{
				minDist = res;
				_class = k + 1;
				classMap[id] = _class;
				local_changes++;
			}
		}
		atomicAdd(changes, local_changes);
	}
}

/* This kernel computes the values needed for the calculation of the new centroids. After the
 * execution, auxCentroids will have stored the sum of all points that belong to each class, and
 * pointPerClass will contain the count of all points belonging to each class */
__global__
void recalculateCentroidsStep1GPU(float* points,	   /* in */	 /* array of points */
								  int* classMap, 	   /* in */	 /* point i belongs to class k */
								  int* pointsPerClass, /* out */ /* num of points for each class k */
								  float* auxCentroids, /* out */ /* new centroids are computed here */
								  int lines,  		   /* in */  /* number of points */
								  int K,			   /* in */  /* number of centroids */
								  int samples)		   /* in */  /* dimensionality of points */
{
	/* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	int _class;
	if (id < lines)
	{
		_class = classMap[id];
		/* Add the values computed locally within each block to the global array. This will be needed
		 * in step 2 */
		atomicAdd(&pointsPerClass[_class - 1], 1);
		for (int j = 0; j < samples; j++)
		{
			/* ATTENTION: There should NOT be an issue with floating point rounding here, since
			 * there shouldn't be values that are too small. Check this. */
			atomicAdd(&auxCentroids[(_class - 1)*samples + j], points[id*samples+j]);
		}
	}
}

/* This kernel divides the sum of the values of all the points belonging to class k by the number
 * of points belonging to class k. */
__global__
void recalculateCentroidsStep2GPU(float* auxCentroids, /* out */  /* new centroids are computed here */
								  int* pointsPerClass, /* in */   /* num of points for each class k */
								  int K,			   /* in */   /* num of classes */
								  int samples)		   /* in */   /* dimensionality of points */
{
	/* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	/* id of a thread inside a block */
	int blockId = threadIdx.x;
	extern __shared__ int sharedPointsPerClass[];

	/* Copy pointsPerClass into shared memory */
	/* ATTENTION: It's better/quicker (for this kernel) to use at most 64 centroids */
	if (K <= BLOCK_DIMX)
	{
		if (blockId < K)
		{
			sharedPointsPerClass[blockId] = pointsPerClass[blockId];
		}
	} else
	{
		sharedPointsPerClass[blockId] = pointsPerClass[blockId];
		if (blockId == 0)
		{
			for (int i = 64; i < K; i++)
			{
				sharedPointsPerClass[i] = pointsPerClass[i];
			}
		}
	}
	__syncthreads(); /* !!!!!!! ATTENTION: Do NOT remove this barrier !!!!!!! */

	if (id < K*samples)
	{
		int k = id / K;
		auxCentroids[id] /= sharedPointsPerClass[k];
	}
}

/* Computes the euclidean distance between point1 and point2 (both of dimensionality samples) */
__device__
float euclideanDistanceGPU(float* point1 /* in  */,   /* Point 1 */
						   float* point2 /* in  */,   /* Point 2 */
						   int samples	 /* in  */)   /* Dimensionality of points */
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++)
	{
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
	}
	return sqrtf(dist);
}

/* This kernel checks whether the distance between the old centroids and the new ones is small
 * enough for us to end the algorithm */
__global__
void recalculateCentroidsStep3GPU(float* maxDist, 		/* out */  /* precision in centroid distance */
								  float* centroids,		/* in */   /* old centroids array */
								  float* auxCentroids,	/* in */   /* new centroids */
								  int samples,			/* in */   /* dimensionality */
								  int K)				/* in */   /* number of centroids */
{
	/* Global thread id */
	int id = threadIdx.x
			 + blockIdx.x * blockDim.x
			 + ( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x * blockDim.x
			 + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	float dist;

	/* ATTENTION: We can speed this up. */
	if (id < K)
	{
		dist = euclideanDistanceGPU(&centroids[id*samples],
									&auxCentroids[id*samples],
									samples);
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
