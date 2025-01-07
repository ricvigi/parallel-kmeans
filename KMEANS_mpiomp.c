/*
 * k-Means clustering algorithm
 *
 * OpenMP version
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
#include <omp.h>
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200
#define NLOGIC_CORES 8


//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* Function declarations */
void showFileError(int error, char* filename);
int readInput(char* filename, int *lines, int *samples);
int readInput2(char* filename, float* data);
int writeResult(int *classMap, int lines, const char* filename);
float euclideanDistance(float *point, float *center, int samples);
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K);
void zeroFloatMatriz(float *matrix, int rows, int columns);
void zeroIntArray(int *array, int size);

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

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++)
		centroidPos[i]=rand()%lines;

	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);

	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j, k;
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
	 * MPI + OMP
	 *
	 */
	int rank, comm_sz;
	int root = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	const MPI_Comm COMM = MPI_COMM_WORLD;

	/* Set the number of threads that are spawned by each MPI thread */
	int nthreads = NLOGIC_CORES / comm_sz;
	omp_set_num_threads(nthreads);

	/* ATTENTION: local_sz is an integer. Check that the division is able to fit the last part of the divided array */
    int local_sz = ceil(lines / comm_sz);

	/* Thread local bufffers */
	float local_data[local_sz*samples];
	int* local_classMap = (int*)calloc(local_sz,sizeof(int));


	// float* local_auxCentroids = (float*)calloc(K*samples, sizeof(float));

	/* Scatter data array and broadcast centroids array. ATTENTION: We might not need to broadcast centroids now */
	if (rank == 0)
	{
		MPI_Scatter(data, local_sz*samples, MPI_FLOAT, local_data, local_sz*samples, MPI_FLOAT, root, MPI_COMM_WORLD);

	} else
	{
		MPI_Scatter(NULL, local_sz*samples, MPI_FLOAT, local_data, local_sz*samples, MPI_FLOAT, root, MPI_COMM_WORLD);
	}


	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	do
	{
		it++;
		changes = 0;
#		pragma omp parallel for shared(local_classMap, changes) private(i, _class, minDist, k, dist)
		for (i = 0; i < local_sz; i++)
		{
			_class = 1;
			minDist = FLT_MAX;
			for (k = 0; k < K; k++)
			{
				dist=euclideanDistance(&local_data[i*samples], &centroids[k*samples], samples);

				if(dist < minDist)
				{
					minDist=dist;
					_class=k+1;
				}
			}
			if(local_classMap[i] != _class)
			{
#				pragma omp atomic
				changes++;
			}
			local_classMap[i]=_class;
		}

		zeroIntArray(pointsPerClass,K);
		zeroFloatMatriz(auxCentroids,K,samples);
#		pragma omp parallel
		{
			int* local_pointsPerClass = (int*) calloc(K,sizeof(int));
			float* local_auxCentroids = (float*) calloc(K*samples,sizeof(float));
#			pragma omp for private (_class, i, j)
			for(int i = 0; i < local_sz; i++)
			{
				/* first step of calculating the mean for each class. Add the value of data
				* points belonging to that class */
				/* ATTENTION: All reduce on local_auxCentroids and pointsPerClass?? most likely YES */
				_class = local_classMap[i];
				local_pointsPerClass[_class-1] += 1;

				for(int j = 0; j < samples; j++)
				{
					local_auxCentroids[(_class-1)*samples+j] += local_data[i*samples+j];
				}
			}


	#		pragma omp critical
			{
				for (int k = 0; k < K; k++)
				{
					pointsPerClass[k] += local_pointsPerClass[k];
					for (int j = 0; j < samples; j++)
					{
						auxCentroids[k * samples + j] += local_auxCentroids[k * samples + j];
					}
				}
				free(local_pointsPerClass);
				free(local_auxCentroids);
			}
		}
		/* int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm c */

		MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_FLOAT, MPI_SUM, COMM);
		MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K*samples, MPI_FLOAT, MPI_SUM, COMM);
		MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, COMM);

		for(k = 0; k < K; k++)
		{
			/* second step of calculating the mean for each class. Divide by the number
		     * of elements of each class */
			for(j = 0; j < samples; j++)
			{
				auxCentroids[k*samples+j] /= pointsPerClass[k];
			}
		}

		/* here we check if maxDist will eventually be bigger than maxThreshold */
		maxDist=FLT_MIN;

		for(k = 0; k < K; k++)
		{
			distCentroids[k] = euclideanDistance(&centroids[k*samples], &auxCentroids[k*samples], samples);
			if(distCentroids[k]>maxDist)
			{
				maxDist=distCentroids[k];
			}
		}

		memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
		MPI_Allreduce(MPI_IN_PLACE, &maxDist, 1, MPI_FLOAT, MPI_MAX, COMM);

		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);
	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));




	end = omp_get_wtime();
	MPI_Gather(local_classMap, local_sz, MPI_INT, classMap, local_sz, MPI_INT, root, COMM);


/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */


	if (rank == 0)
{

		// MPI_Reduce(
	//    void* send_data,
	//    void* recv_data,
	//    int count,
	//    MPI_Datatype datatype,
	//    MPI_Op op,
	//    int root,
	//    MPI_Comm communicator)
	MPI_Reduce(MPI_IN_PLACE, &start, 1, MPI_DOUBLE, MPI_MIN, root, COMM);
	MPI_Reduce(MPI_IN_PLACE, &end, 1, MPI_DOUBLE, MPI_MAX, root, COMM);
	// Output and termination conditions
	printf("%s",outputMsg);

	//END CLOCK*****************************************

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

	free(local_classMap);
	// free(local_auxCentroids);

	MPI_Finalize();
	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
} else
{
	MPI_Reduce(&start, NULL, 1, MPI_DOUBLE, MPI_MIN, root, COMM);
	MPI_Reduce(&end, NULL, 1, MPI_DOUBLE, MPI_MAX, root, COMM);


	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	free(local_classMap);
	// free(local_auxCentroids);

	MPI_Finalize();
}
	return 0;
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

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}

