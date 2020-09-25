#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network_state state);
void backward_logistic_layer(const layer l, network_state state);

#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network_state state);
void backward_logistic_layer_gpu(const layer l, network_state state);
#endif

#endif
