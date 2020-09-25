#ifndef RYOLO_LAYER_H
#define RYOLO_LAYER_H
//#include "darknet.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_Ryolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
Rbox get_Ryolo_box(float* x, float* biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
void forward_Ryolo_layer(const layer l, network_state state);
void backward_Ryolo_layer(const layer l, network_state state);
void resize_Ryolo_layer(layer *l, int w, int h);
int Ryolo_num_detections(layer l, float thresh);
int Ryolo_num_detections_batch(layer l, float thresh, int batch);
int get_Ryolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int get_Ryolo_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch);
float Rbox_iou(Rbox a, Rbox b);
float Rbox_union(Rbox a, Rbox b);
float Rbox_intersection(Rbox a, Rbox b);
float overlap_Ryolo(float x1, float w1, float x2, float w2);
void correct_Ryolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_Ryolo_layer_gpu(const layer l, network_state state);
void backward_Ryolo_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif