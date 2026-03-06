#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Multi(2)-layer perceptron for XOR.
 *      
 *   in  -> hidden layer -> out 
 *   (2) ->    (2)       -> (1)
 -----------------------------------
 *             (h1)             
 *  [x] ==>     |       ==> [y] 
 *             (h2)             
 -----------------------------------
 */

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define SAMPLES 4

/* Training hyperparameters */
#define LEARNING_RATE 0.5
#define EPOCHS 500

typedef struct {
    /* Input -> Hidden */
    double W1[HIDDEN_SIZE][INPUT_SIZE];
    double b1[HIDDEN_SIZE];

    /* Hidden -> Output */
    double W2[HIDDEN_SIZE];
    double b2;
} Perceptron2Layer;

/* Sigmoid activation and its derivative */
static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x));}

/* If y = sigmoid(x), then derivative is y * (1 - y) */
static double sigmoid_derivative_from_output(double y) { return y * (1.0 - y);}

/* Small helper for random weight initialization in range [-1, 1] */
static double rand_weight(void) { return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;}

// Initialize perceptron
static void init_perceptron(Perceptron2Layer *model, unsigned int seed) {
    srand(seed);
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            model->W1[h][i] = rand_weight();
        }
        model->b1[h] = rand_weight();
        model->W2[h] = rand_weight();
    }
    model->b2 = rand_weight();
}

// Forward propagation:
static void forward_pass(
    const Perceptron2Layer *model,
    const double x[INPUT_SIZE],
    double hidden[HIDDEN_SIZE],
    double *y_hat
) {
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        double z = model->b1[h];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            z += model->W1[h][i] * x[i];
        }
        hidden[h] = sigmoid(z);
    }

    double z_out = model->b2;
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        z_out += model->W2[h] * hidden[h];
    }
    *y_hat = sigmoid(z_out);
}

/*
 * Train on one sample and return 0.5 * (prediction - target)^2.
 * This function contains the core 4 training steps:
 * 1) forward pass
 * 2) output error
 * 3) hidden backprop
 * 4) parameter updates
 */
static double train_one_sample(
    Perceptron2Layer *model,
    const double x[INPUT_SIZE],
    double y_target
) {
    double hidden[HIDDEN_SIZE];
    double y_hat = 0.0;
    double old_W2[HIDDEN_SIZE];

    /* 1) Forward pass */
    forward_pass(model, x, hidden, &y_hat);

    /* 2) Output error (sigmoid + BCE gives: delta_out = y_hat - y_target) */
    double delta_out = y_hat - y_target;

    /* 3) Backprop to hidden layer (use old output weights for clarity) */
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        old_W2[h] = model->W2[h];
    }
    double delta_hidden[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        delta_hidden[h] =
            (old_W2[h] * delta_out) * sigmoid_derivative_from_output(hidden[h]);
    }

    /* 4) Gradient descent update */
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        model->W2[h] -= LEARNING_RATE * delta_out * hidden[h];
    }
    model->b2 -= LEARNING_RATE * delta_out;

    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            model->W1[h][i] -= LEARNING_RATE * delta_hidden[h] * x[i];
        }
        model->b1[h] -= LEARNING_RATE * delta_hidden[h];
    }

    {
        double diff = y_hat - y_target;
        return 0.5 * diff * diff;
    }
}

static double predict(const Perceptron2Layer *model, const double x[INPUT_SIZE]) {
    double hidden[HIDDEN_SIZE];
    double y_hat = 0.0;
    forward_pass(model, x, hidden, &y_hat);
    return y_hat;
}

int main(void) {
    /*
     * XOR truth table:
     * x1 x2 -> y
     * 0  0  -> 0
     * 0  1  -> 1
     * 1  0  -> 1
     * 1  1  -> 0
     */
    const double X[SAMPLES][INPUT_SIZE] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    const double Y[SAMPLES] = {0.0, 1.0, 1.0, 0.0};

    Perceptron2Layer model;
    init_perceptron(&model, 42);

    /* Stochastic gradient descent over the 4 XOR samples */
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss = 0.0;
        for (int s = 0; s < SAMPLES; ++s) {
            epoch_loss += train_one_sample(&model, X[s], Y[s]);
        }

        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %5d | avg loss = %.6f\n", epoch + 1, epoch_loss / SAMPLES);
        }
    }

    /* Final predictions after training */
    printf("\nFinal predictions for XOR:\n");
    for (int s = 0; s < SAMPLES; ++s) {
        double y_hat = predict(&model, X[s]);
        printf("Input: [%.0f, %.0f]  Target: %.0f  Pred: %.4f\n",
               X[s][0], X[s][1], Y[s], y_hat);
    }

    return 0;
}
