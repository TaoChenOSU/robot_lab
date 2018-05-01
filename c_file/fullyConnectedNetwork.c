/*
**  C implementation of a fully connected network
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

int main(void){
  int i, j;  // counter
  // To load the images and labels
  int *test_labels;
  double **test_images;
  double **previous_output, **results;
  double loss;
  printf("Obtaining data...\n");
  int num_images = load_data(&test_labels, &test_images);
  if(num_images == -1){
    return 1;
  }
  /*
  * The network in the assignment has 6 layers of fully-connected layer,
  * each of them is combined with a relu activation layer.
  */
  struct FullyConnectLayer fc1, fc2, fc3, fc4, fc5, fc6;

  // layer 1
  layer_init(&fc1, "fc1", 3072, 100);
  // layer 2
  layer_init(&fc2, "fc2", 100, 100);
  // layer 3
  layer_init(&fc3, "fc3", 100, 100);
  // layer 4
  layer_init(&fc4, "fc4", 100, 100);
  // layer 5
  layer_init(&fc5, "fc5", 100, 100);
  // layer 6
  layer_init(&fc6, "fc6", 100, 10);
  //
  previous_output = forward_pass(&fc1, test_images, num_images, 3072);
  relu_activatioin(previous_output, num_images, 100);
  previous_output = forward_pass(&fc2, previous_output, num_images, 100);
  relu_activatioin(previous_output, num_images, 100);
  previous_output = forward_pass(&fc3, previous_output, num_images, 100);
  relu_activatioin(previous_output, num_images, 100);
  previous_output = forward_pass(&fc4, previous_output, num_images, 100);
  relu_activatioin(previous_output, num_images, 100);
  previous_output = forward_pass(&fc5, previous_output, num_images, 100);
  relu_activatioin(previous_output, num_images, 100);
  results = forward_pass(&fc6, previous_output, num_images, 100);

  loss = cross_entropy(results, test_labels, num_images, 10);
  printf("The loss is %.15f\n", loss);
}
