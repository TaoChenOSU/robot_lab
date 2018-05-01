/*
**  C implementation of a fully connected network
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "structs.h"

#define TRUE 0
#define FALSE 1

/* Assign weights to a layer
** inputs: the pointer to the layer, the weight and the weight's diemnsion
** outputs: none
*/
void assign_weights(struct FullyConnectLayer *fc, double **weight, int height, int width);

/* Assign bias to a layer
** inputs: the pointer to the layer, the bias and the bias's diemnsion
** outputs: none
*/
void assign_bias(struct FullyConnectLayer *fc, double *bias, int width);

/* The forward pass function.
** inputs: a layer, and the data, data dimension
** outputs: a weighted data plus bias
*/
double **forward_pass(struct FullyConnectLayer const *fc, double **data, int height, int width);

/* relu activation function
** inputs: output from the previous layer, the dimension
** outputs: none, directly modify the original flattened matrix
*/
void relu_activatioin(double **inputs, int batch_size, int output_dim);

/* load data from all the trainning data set and get the
** mean to do normalization.
*/
double *mean_data();

/* load data from the test batch., which contains 10,000 images.
** store the data in two matrix: labels, images.
*/
int load_data(int **labels, double ***images);

/* Get the model weight from files output from python code
** input: layer name
** output: a formatted 2D array
*/
double **get_model_weight(char *layer_name, int input_dim, int output_dim);

/* Get the model bias from files output from python code
** input: layer name
** output: a formatted array
*/
double *get_model_bias(char *layer_name, int output_dim);

/* intialize a layer, including getting the weights and bias
** from files.
*/
void layer_init(struct FullyConnectLayer *fc, char *name, int input_dim, int output_dim);

/* calculate the loss of the results
** inputs: results and correct labels, batch size and number of classes
** outputs: loss
*/
double cross_entropy(double **results, int *labels, int batch_size, int num_classes);
