#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#pragma pack(1)

typedef struct link_t link_t;

typedef struct node_t {
    int id;
    int outcon_size;
    float cdela;
    int padding;
    float input;
    float output;
    link_t * outcon;
}node_t;

typedef struct link_t {
    int id;
    float weight;
    node_t * from;
    node_t * to;
}link_t;

typedef struct model_t {
    node_t * nodes;
    int nodes_size;
    int bias_id;
}model_t;

float randf(float h, float l) {
    float s = rand() / (float) RAND_MAX;
    return l + s * (h - l);
}

void make_ann(model_t * model, int layer_size, int layer[layer_size]) {
    assert(model != NULL);
    int total_nodes=0;
    model->bias_id = layer[0]; // bias node in input layer, store it's id
    layer[0] += 1; // increase input layer size by one
    for(int i=layer_size-1; i >= 0; --i) {
        total_nodes += layer[i];
    }
    puts("making ann");
    puts("-----------------");
    printf("total nodes:%d\n", total_nodes);
    model->nodes = malloc(sizeof(node_t) * total_nodes);
    model->nodes_size = total_nodes;
    node_t *pnode = model->nodes;

    int node_id = 0;
    int link_id = 0;

    for(int l=0; l<layer_size-1; l++) {
        for(int i=0; i<layer[l]; i++) {
            pnode[i].input = 0.0f;
            pnode[i].output = 0.0f;
            pnode[i].id = node_id++;
            pnode[i].outcon = malloc(sizeof(link_t) * layer[l+1]); // each node holds independent links obj
            pnode[i].outcon_size = layer[l+1];
            for(int j=0; j<layer[l+1]; j++) {
                pnode[i].outcon[j].from = pnode + i;
                pnode[i].outcon[j].to = pnode + layer[l] + j;
                pnode[i].outcon[j].weight = randf(-100.0f, 100.0f);
                pnode[i].outcon[j].id = link_id++;
            }
        }
        pnode+=layer[l]; // with every layer, starting change
    }

    for(int i = 0; i < layer[layer_size-1]; i++) {
        pnode[i].outcon = NULL;
        pnode[i].id = node_id++;
        pnode[i].input = 0.0f;
        pnode[i].output = 0.0f;
    }
    printf("node_cnt:%d  link_cnt:%d\n", node_id, link_id);
}

void print_ann(model_t model) {
    for(int i=0; i<model.nodes_size; i++) {
        for(int j=0; j<model.nodes[i].outcon_size; j++) {
            printf("%d: %d * %f * %d\n", model.nodes[i].outcon[j].id, model.nodes[i].outcon[j].from->id,model.nodes[i].outcon[j].weight ,model.nodes[i].outcon[j].to->id);
        }
    }
    printf("bias_node at:%d\n", model.bias_id);
}

void print_node_state(model_t model) {
    for(int i=0; i<model.nodes_size; i++) {
        printf("%d: %f _ %f\n", model.nodes[i].id, model.nodes[i].input, model.nodes[i].output);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void feed_forward(model_t model, int intput_size, float input[intput_size], int output_size, float* output) {
    for(int i = 0; i < model.nodes_size; i++) {
        model.nodes[i].input = 0.0f;
    }
    model.nodes[model.bias_id].input = 1.0f;
    for(int i = 0; i < intput_size; i++) {
        model.nodes[i].input = input[i];
    }
    for(int i = 0; i < model.nodes_size; i++) {
        model.nodes[i].output = sigmoid(model.nodes[i].input);
        for(int l=0; l < model.nodes[i].outcon_size; l++) {
            model.nodes[i].outcon[l].to->input += model.nodes[i].output * model.nodes[i].outcon[l].weight;
        }
    }
    for(int i = 0; i < output_size; i++) {
        output[i] = model.nodes[model.nodes_size - output_size + i].output;
    }
}

void feed_backward(model_t model, int err_size, float err[err_size]) {
    for(int i = 0; i < err_size; i++) {
        model.nodes[model.nodes_size - err_size + i].cdela = 2.0f *(err[i]);
    }
    for(int i = model.nodes_size-err_size-1; i>=0; i--) {
        model.nodes[i].cdela = 0.0f;
        for(int j=0; j<model.nodes[i].outcon_size; j++) {
            float t = model.nodes[i].outcon[j].to->output;
            model.nodes[i].cdela += model.nodes[i].outcon[j].to->cdela * t * (1.0f - t)* model.nodes[i].outcon[j].weight;
        }
    }
    for(int i=model.nodes_size-err_size-1; i>=0; i--) {
        for(int j=0; j<model.nodes[i].outcon_size; j++) {
            float t = model.nodes[i].outcon[j].to->output;
            t = model.nodes[i].outcon[j].to->cdela * t * (1.0f - t);
            printf("%f   ", t);
            model.nodes[i].outcon[j].weight -= t;
        }
    }
}




int main() {
    srand(42);
    model_t model;
    int layers[] = {2, 3, 1};
    make_ann(&model, 3, layers);
    print_ann(model);

    float input[4][2] = {
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f}
    };

    float result[4] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    float buffer[1];
    float error_term[1];

    for(int i = 0; i < 100; i++) {
        for(int k =0; k < 1; k++) error_term[k] = 0.0f;
        for(int j=0; j < 4; j++) {
            feed_forward(model, 2, (float*)&(input[j]), 1, (float *)&buffer);
            float tmp;
            for(int k =0; k < 1; k++) {
                tmp = buffer[k] - result[j];
                error_term[j] = tmp;
            }
        }
        for(int j=0; j < 4; j++) {
                error_term[j] /= 4.0f;
        }
        feed_backward(model, 1, (float *)&error_term);
    }
    
    print_ann(model);

    for(int j=0; j < 4; j++) {
        feed_forward(model, 2, (float*)&(input[j]), 1, (float *)&buffer);
        for(int k=0; k < 1; k++) {
            printf("%f   ", buffer[k]);
        }
        puts(" ");
    }

    return 0;
};