/*
**  C implementation of a fully connected network
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

/* Assign weights to a layer
** inputs: the pointer to the layer, the weight and the weight's diemnsion
** outputs: none
*/
void assign_weights(struct FullyConnectLayer *fc, double **weight, int height, int width){
  int i, j; // counters
  if(fc->input_dim != height || fc->output_dim != width){
    fprintf(stderr, "In assigning weight, dimensions don't match.\n");
    exit(1);
  }
  fc->weight = malloc(sizeof(double*)*height);
  for(i = 0; i < height; i++){
    fc->weight[i] = malloc(sizeof(double)*width);
    for(j = 0; j < width; j++){
      fc->weight[i][j] = weight[i][j];
    }
  }
}

/* Assign bias to a layer
** inputs: the pointer to the layer, the bias and the bias's diemnsion
** outputs: none
*/
void assign_bias(struct FullyConnectLayer *fc, double *bias, int width){
  int i; // counter
  if(fc->output_dim != width){
    fprintf(stderr, "In assigning bias, dimension doesn't match.\n");
    exit(1);
  }
  fc->bias = malloc(sizeof(double)*width);
  for(i = 0; i < width; i++){
    fc->bias[i] = bias[i];
  }
}

/* The forward pass function.
** inputs: a layer, and the data, data dimension
** outputs: a weighted data plus bias
*/
double **forward_pass(struct FullyConnectLayer const *fc, double **data, int batch_size, int input_dim){
  if(fc->input_dim != input_dim){
    fprintf(stderr, "%s: ", fc->name);
    fprintf(stderr, "data dimension doesn't match the layer input dimension.\n");
    exit(1);
  }
  int i, j, k; // counters

  // allocate memory for the output matrix
  double **layer_output = malloc(sizeof(double*)*batch_size);
  for(i = 0; i < batch_size; i++){
    layer_output[i] = malloc(sizeof(double)*fc->output_dim);
  }
  // matirx multiplication
  for(i = 0; i < batch_size; i++){
    for(j = 0; j < fc->output_dim; j++){
      layer_output[i][j] = 0;
      for(k = 0; k < fc->input_dim; k++){
        layer_output[i][j] += data[i][k] * fc->weight[k][j];
      }
      layer_output[i][j] += fc->bias[j];
    }
  }

  return layer_output;
}

/* relu activation function
** inputs: output from the previous layer, the dimension
** outputs: none, directly modify the original flattened matrix
*/
void relu_activatioin(double **inputs, int batch_size, int output_dim){
  int i, j;

  for(i = 0; i < batch_size; i++){
    for(j = 0; j < output_dim; j++){
      if(inputs[i][j] <= 0)
        inputs[i][j] = 0.0;
    }
  }
}

/* load data from all the trainning data set and get the
** mean to do normalization.
*/
double *mean_data(){
  int i, j, k; // counters
  FILE *fp;
  size_t result;
  int images_per_file = 10000;
  double mean;
  double *means = malloc(sizeof(double)*3072);
  int images_to_get[] = {10000, 10000, 10000, 10000, 9000};
  int trainning_image_size = 49000;
  double **images;
  unsigned char image_buffer[3073];
  const char *file_paths[5] = {
    "../data/cifar-10-batches-bin/data_batch_1.bin",
    "../data/cifar-10-batches-bin/data_batch_2.bin",
    "../data/cifar-10-batches-bin/data_batch_3.bin",
    "../data/cifar-10-batches-bin/data_batch_4.bin",
    "../data/cifar-10-batches-bin/data_batch_5.bin"
  };

  // allocate memory so that can be freed later
  images = malloc(sizeof(double*)*trainning_image_size);
  for(i = 0; i < trainning_image_size; i++){
    images[i] = malloc(sizeof(double)*3072);
  }

  for(i = 0; i < 5; i++){
    fp = fopen(file_paths[i], "rb");
    for(j = 0; j < images_to_get[i]; j++){
      fseek(fp, 0, j*3073);
      result = fread(image_buffer, 1, sizeof(image_buffer), fp);
      if(result != 3073){
        fprintf(stderr, "Error in reading data: unaligned bytes.\n");
        exit(1);
      }
      for(k = 0; k < 3072; k++){
        images[images_per_file*i+j][k] = image_buffer[k+1];
      }
    }
    fclose(fp);
  }

  // compute the mean
  for(i = 0; i < 3072; i++){
    mean = 0;
    for(j = 0; j < trainning_image_size; j++){
      mean += images[j][i];
    }
    means[i] = mean/trainning_image_size;
  }
  free(images);
  return means;
}

/* load data from the test batch., which contains 10,000 images.
** store the data in two matrix: labels, images.
*/
int load_data(int **labels, double ***images){
  int i, j; // counters
  size_t result;
  double *normalize_factor;
  unsigned char image_buffer[3073];
  const char *file_path = "../data/cifar-10-batches-bin/test_batch.bin";
  FILE *fp = fopen(file_path, "rb");
  // get the normalize factor from training files
  normalize_factor = mean_data();

  // Get the file size to allocate memory for labels and images
  long file_size;
  // define how many samples you want to drwa from the file
  int num_images = 1000;
  if(num_images == 0){
    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    num_images = file_size/3073;
  }

  *labels = malloc(sizeof(int)*num_images);
  *images = malloc(sizeof(double*)*num_images);
  // Rewind back to the beginning of the file
  rewind(fp);
  for(i = 0; i < num_images; i++){
    fseek(fp, 0, i*3073);
    result = fread(image_buffer, 1, sizeof(image_buffer), fp);
    if(result != 3073){
      fprintf(stderr, "Error in reading data: unaligned bytes.\n");
      exit(1);
    }
    (*labels)[i] = image_buffer[0];
    (*images)[i] = malloc(sizeof(double)*3072);
    for(j = 0; j < 3072; j++){
      (*images)[i][j] = image_buffer[j+1];
      // normalize the images
      (*images)[i][j] -= normalize_factor[j];
    }
  }
  fclose(fp);
  return num_images;
}

/* Get the model weight from files output from python code
** input: layer name
** output: a formatted 2D array
*/
double **get_model_weight(char *layer_name, int input_dim, int output_dim){
  int i, j; // counters
  double **weights;
  char buffer[5000];
  char *token;

  const char parent_dir[] = "../sgdm_params/";
  const char extension[] = "_w.txt";
  char file_path[24];

  strcpy(file_path, parent_dir);
  strcat(file_path, layer_name);
  strcat(file_path, extension);
  FILE *fp = fopen(file_path, "r");
  if(fp == NULL){
    fprintf(stderr, "Error in opening weight files");
    exit(1);
  }
  // allocate space for the output
  weights = malloc(sizeof(double*)*input_dim);
  for(i = 0; i < input_dim; i++){
    weights[i] = malloc(sizeof(double)*output_dim);
  }

  i = 0;
  while(fgets(buffer, sizeof(buffer), fp) != NULL){
    weights[i][0] = atof(strtok(buffer, ","));
    for(j = 1; j < output_dim; j++){
      weights[i][j] = atof(strtok(NULL, ","));
    }
    i++;
  }
  fclose(fp);
  return weights;
}

/* Get the model bias from files output from python code
** input: layer name
** output: a formatted array
*/
double *get_model_bias(char *layer_name, int output_dim){
  int i, j; // counters
  double *bias;
  char buffer[100];

  const char parent_dir[] = "../sgdm_params/";
  const char extension[] = "_b.txt";
  char file_path[24];

  strcpy(file_path, parent_dir);
  strcat(file_path, layer_name);
  strcat(file_path, extension);
  FILE *fp = fopen(file_path, "r");
  if(fp == NULL){
    fprintf(stderr, "Error in opening bias files");
    exit(1);
  }
  // allocate space for the output
  bias = malloc(sizeof(double)*output_dim);

  i = 0;
  while(fgets(buffer, sizeof(buffer), fp) != NULL){
    bias[i] = atof(buffer);
    i++;
  }
  fclose(fp);
  return bias;
}

/* intialize a layer, including getting the weights and bias
** from files.
*/
void layer_init(struct FullyConnectLayer *fc, char *name, int input_dim, int output_dim){
  fc->name = name;
  fc->input_dim = input_dim;
  fc->output_dim = output_dim;
  fc->weight = get_model_weight(fc->name, input_dim, output_dim);
  fc->bias = get_model_bias(fc->name, output_dim);
  fprintf(stdout, "Finished initializing %s layer.\n", fc->name);
}

/* calculate the loss of the results
** inputs: results and correct labels, batch size and number of classes
** outputs: loss
*/
double cross_entropy(double **results, int *labels, int batch_size, int num_classes){
  int i, j; // counters
  double adjusted_results[batch_size][num_classes]; // adjust result after sotfmax
  double sum;
  double loss;

  // sotfmax
  for(i = 0; i < batch_size; i++){
    sum = 0;
    for(j = 0; j < num_classes; j++){
      adjusted_results[i][j] = exp(results[i][j]);
      sum += adjusted_results[i][j];
    }
    for(j = 0; j < num_classes; j++){
      adjusted_results[i][j] = adjusted_results[i][j]/sum;
    }
  }

  // cross entropy
  loss = 0;
  for(i = 0; i < batch_size; i++){
    loss += log(adjusted_results[i][labels[i]]);
  }
  loss = -loss/batch_size;
  return loss;
}
