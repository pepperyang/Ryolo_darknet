#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#include "dark_cuda.h"//Syolo
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_logistic_layer(int batch, int inputs)
{
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.truths = inputs;
    l.loss = xcalloc(inputs*batch, sizeof(float));
    l.output = xcalloc(inputs*batch, sizeof(float));
    l.delta = xcalloc(inputs*batch, sizeof(float));
    l.cost = xcalloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;
    #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}

void forward_logistic_layer(const layer l, network_state state)//network net
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    if(state.truth){
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, state.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_logistic_layer(const layer l, network_state state)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_logistic_layer_gpu(const layer l, network_state state)
 {
    int i, j, k;
    simple_copy_ongpu(l.outputs * l.batch, state.input, l.output_gpu);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, LOGISTIC);
    //cuda_pull_array(l.output_gpu, state.input, l.batch*l.inputs);


    float* in_cpu = (float*)xcalloc(l.batch * l.outputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    memcpy(in_cpu, l.output, l.batch * l.outputs * sizeof(float));
    float* truth_cpu = 0;
    float loss = 0;
    if (state.truth) {
        int num_truth = l.batch * l.truths;
        truth_cpu = (float*)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);

        //loss = dice_loss_cpu(l.batch * l.inputs, l.output, truth_cpu, l.delta, l.loss);
        //*(l.cost) = loss;
        logistic_x_ent_cpu(l.batch * l.inputs, l.output, truth_cpu, l.delta, l.loss);
        *(l.cost) = mean_array(l.loss, l.batch * l.inputs)* l.batch;
    }

    //cuda_pull_array(net.truth_gpu, net.truth, l.batch*l.inputs);
    //image im = make_image(1024, 512, 1);
    //image im_truth = make_image(1024, 512, 1);
    //for(i=0; i<l.w*l.h; i++){
      //l.delta[i] = truth_cpu[i] - l.output[i];
      //im.data[i] = (float)net.input[i];
      //im_truth.data[i] = (float)net.truth[i];
   // }
    //cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //save_image(im, "feature_map");
    //save_image(im_truth, "truth");
    //free_image(im);
    //free_image(im_truth);

    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    free(in_cpu);
    free(truth_cpu);


}

void backward_logistic_layer_gpu(const layer l, network_state state)
{
    //axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    axpy_ongpu(l.batch * l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}

#endif
