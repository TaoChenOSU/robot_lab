/*
Assignment: Byte-level communications.

Suppose we have trained our model in TensorFlow,
and we're ready to try it on a real quadrotor.
We need to send the trained model over the radio to the quadrotor.

Your job: implement the send_weights() function by using the send() function,
and the recv_weights() function by using the recv() function.

Note that /onboard_weights/ is bigger than /pc_weights/, because it's designed to allow
changes in the size of the neural network without recompiling the onboard code.
So you need to communicate the actual neural network size over the radio.
(Your recv_weights() function should not reference the IN_SIZE and OUT_SIZE macros.)

Your read_weights() should fill the onboard_in_size and onboard_out_size variables
and the onboard_weights array.
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// --------------------------------------------------------------
//            Do not modify the code inside this box.

// the trained neural network weights on the PC.
#define IN_SIZE 12
#define OUT_SIZE 20
static float pc_weights[OUT_SIZE][IN_SIZE];

static uint8_t radio_buffer[1000 + 32 * 32 * sizeof(float)];
static int send_idx = 0;
static int recv_idx = 0;

static uint8_t onboard_in_size = 0;
static uint8_t onboard_out_size = 0;
static float onboard_weights[32][32];

// request to send up to /count/ bytes over the radio from the array /bytes/.
// returns the actual number of bytes sent, may be lower than requested.
int send(uint8_t const *bytes, int count)
{
	int n = rand() % (count + 1);
	for (int i = 0; i < n; ++i) {
		radio_buffer[send_idx++] = bytes[i];
	}
	return n;
}

// request to receive up to /count/ bytes over the radio and store into /bytes/.
// returns the actual number of bytes receiveuint8_t dimensions[2] = {OUT_SIZE, IN_SIZE};
int recv(uint8_t *bytes, int count)
{
	int n = rand() % (count + 1);
	if (recv_idx + n > send_idx) n = send_idx - recv_idx;
	for (int i = 0; i < n; ++i) {
		bytes[i] = radio_buffer[recv_idx++];
	}
	return n;
}
// --------------------------------------------------------------

void send_weights()
{
	int i, j, k;	// counters
	uint8_t *weight;
	// send the dimensions at the beginning
	uint8_t width = IN_SIZE;
 	uint8_t height = OUT_SIZE;
	while(send(&width, 1) != 1) continue;
	while(send(&height, 1) != 1) continue;

	// send the weights byte by byte
	for(i = 0; i < OUT_SIZE; i++){
		for(j = 0; j < IN_SIZE; j++){
			weight = (uint8_t*)(&pc_weights[i][j]);
			for(k = 0; k < (int)sizeof(float); k++){
				while(send(&weight[k], 1) != 1) continue;
			}
		}
	}
}

void recv_weights()
{
	// a union to convert bytes to a float
	union{
		float num;
		uint8_t bytes[sizeof(float)];
	} byte_to_float;

	int i, j, k;	// counters
	uint8_t width, height;
	// get the dimension first
	while(recv(&width, 1) != 1) continue;
	while(recv(&height, 1) != 1) continue;
	onboard_in_size = width;
	onboard_out_size = height;

	// recieve the weights number by number
	for(i = 0; i < height; i++){
		for(j = 0; j < width; j++){
			for(k = 0; k < (int)sizeof(float); k++){
				while(recv(&byte_to_float.bytes[k], 1) != 1) continue;
			}
			onboard_weights[i][j] = byte_to_float.num;
		}
	}
}

int main()
{
	srand(time(0));
	// initialize the weights array on the PC.
	for (int i = 0; i < OUT_SIZE; ++i) {
		for (int j = 0; j < IN_SIZE; ++j) {
			pc_weights[i][j] = 0.001 * (rand() - (RAND_MAX / 2.0));
		}
	}

	// transmit the weights.
	send_weights();
	puts("send complete.");
	recv_weights();
	puts("recv complete.");

	// test.
	if (onboard_in_size != IN_SIZE) {
		printf("error: onboard_in_size %d is wrong\n", onboard_in_size);
		return 1;
	}
	if (onboard_out_size != OUT_SIZE) {
		printf("error: onboard_out_size %d is wrong\n", onboard_out_size);
		return 1;
	}
	for (int i = 0; i < OUT_SIZE; ++i) {
		for (int j = 0; j < IN_SIZE; ++j) {
			if (onboard_weights[i][j] != pc_weights[i][j]) {
				printf("error: weight[%d][%d] not equal: PC = %f, onboard = %f\n",
					i, j, pc_weights[i][j], onboard_weights[i][j]);
				return 1;
			}
		}
	}

	puts("weights transmitted correctly!");
	return 0;
}
