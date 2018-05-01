/*
**  C implementation of a fully connected network
*/

struct FullyConnectLayer{
  char *name;
  int input_dim, output_dim;
  double **weight;
  double *bias;
};
