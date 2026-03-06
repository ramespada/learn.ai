#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_FILE "weights.txt"
#define MAX_GATE_LABEL 4
#define MAX_MODELS 128
#define LEARNING_RATE 0.1f

/*
===============================
 SINGLE LAYER PERCEPTRON (SLP)
===============================
*/

/* One-neuron perceptron state: two input weights and a bias term. */
typedef struct {
    float w1;
    float w2;
    float bias;
    float learning_rate;
} Perceptron;

/* One saved model associated with a gate label (and/or/xor/etc.). */
typedef struct {
    char gate[MAX_GATE_LABEL];
    Perceptron p;
} ModelEntry;

static int is_binary_value(int value) {
    return value == 0 || value == 1;
}

static int parse_ok_response(char response, int *is_ok) {
    if (response == 'y' || response == 'Y') {
        *is_ok = 1;
        return 1;
    }
    if (response == 'n' || response == 'N') {
        *is_ok = 0;
        return 1;
    }
    return 0;
}

static void discard_stdin_line(void) {
    int ch;
    do {
        ch = getchar();
    } while (ch != '\n' && ch != EOF);
}

/*
 * Normalize and validate gate labels.
 * Allowed chars: letters, digits, '_' and '-'.
 * Stored normalized as lowercase.
 */
static int normalize_gate_label(const char *in, char out[MAX_GATE_LABEL]) {
    size_t len;
    size_t i;

    if (!in) {return 0;}
    len = strlen(in);
    if (len == 0 || len >= MAX_GATE_LABEL) {return 0;}

    for (i = 0; i < len; i++) {
        unsigned char c = (unsigned char)in[i];
        if (!(isalnum(c) || c == '_' || c == '-')) {
            return 0;
        }
        out[i] = (char)tolower(c);
    }

    out[len] = '\0';
    return 1;
}

/* Step activation with threshold 0. */
static int predict(const Perceptron *p, int x1, int x2) {
    float sum = p->w1 * x1 + p->w2 * x2 + p->bias; // integration
    return (sum >= 0.0f) ? 1 : 0;                  // step-function
}

/* Perceptron learning rule update. */
static void update_weights(Perceptron *p, int x1, int x2, int target, int output) {
    int error = target - output;
    p->w1   += p->learning_rate * error * x1;
    p->w2   += p->learning_rate * error * x2;
    p->bias += p->learning_rate * error     ;
    
    printf("Updated weights: w1=%.4f, w2=%.4f, bias=%.4f\n",p->w1, p->w2, p->bias);

}

/* Find model index by gate label; returns -1 if not found. */
static int find_model(const ModelEntry *entries, int count, const char *gate) {
    int i;
    for (i = 0; i < count; i++) {
        if (strcmp(entries[i].gate, gate) == 0) {
            return i;
        }
    }
    return -1;
}

/* Load all saved models from disk. Missing file is treated as empty model list. */
static int load_models(ModelEntry entries[MAX_MODELS], int *count) {
    FILE *f;
    char line[256];

    *count = 0;
    f = fopen(MODEL_FILE, "r");
    if (!f) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        ModelEntry e;
        char gate_raw[MAX_GATE_LABEL];
        char gate_norm[MAX_GATE_LABEL];

        if (line[0] == '\n' || line[0] == '#') {
            continue;
        }

        if (sscanf(line, "%31s %f %f %f %f", gate_raw, &e.p.w1, &e.p.w2, &e.p.bias, &e.p.learning_rate) != 5) {
            continue;
        }

        if (!normalize_gate_label(gate_raw, gate_norm)) {
            continue;
        }

        if (*count >= MAX_MODELS) {
            fclose(f);
            return 0;
        }

        strcpy(e.gate, gate_norm);
        entries[*count] = e;
        (*count)++;
    }

    fclose(f);
    return 1;
}

/* Save every gate model to disk (one line per gate label). */
static int save_models(const ModelEntry *entries, int count) {
    FILE *f;
    int i;

    f = fopen(MODEL_FILE, "w");
    if (!f) {
        return 0;
    }

    fprintf(f, "# gate w1 w2 bias learning_rate\n");
    for (i = 0; i < count; i++) {
        fprintf(f, "%s %.8f %.8f %.8f %.8f\n",
                entries[i].gate,
                entries[i].p.w1,
                entries[i].p.w2,
                entries[i].p.bias,
                entries[i].p.learning_rate);
    }

    fclose(f);
    return 1;
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s train <gate_label>\n", prog);
    printf("  %s run <gate_label> <x1> <x2>\n", prog);
    printf("\n");
    printf("Examples:\n");
    printf("  %s train and\n", prog);
    printf("  %s run and 1 0\n", prog);
    printf("\n");
    printf("Fixed learning rate: %.3f\n", LEARNING_RATE);
}

/*
 * Interactive, user-supervised training:
 * 1) user provides inputs
 * 2) model predicts
 * 3) user confirms if prediction is correct
 * 4) target is inferred, then model updates from prediction error
 */
static void train_mode(const char *gate) {
    ModelEntry entries[MAX_MODELS];
    int count;
    int idx;
    Perceptron p;
    int iteration;

    if (!load_models(entries, &count)) {
        printf("Could not read model storage from %s\n", MODEL_FILE);
        return;
    }

    idx = find_model(entries, count, gate);
    if (idx >= 0) {
        p = entries[idx].p;
        p.learning_rate = LEARNING_RATE;
        printf("Continuing training for gate '%s'\n", gate);
    } else {
        p.w1 = 0.0f;
        p.w2 = 0.0f;
        p.bias = 0.0f;
        p.learning_rate = LEARNING_RATE;
        printf("Starting new model for gate '%s'\n", gate);
    }

    printf("Type inputs as two integers: x1 x2 (each 0 or 1).\n");
    printf("Type -1 -1 to quit.\n");
    //print_weights(&p);
    printf("weights: w1=%.4f, w2=%.4f, bias=%.4f\n",p.w1, p.w2, p.bias);

    iteration = 1;
    while (1) {
        int read_items;
        int x1;
        int x2;
        int output;
        char response;
        int is_ok;
        int target;

        printf("\niter %d > ", iteration);
        read_items = scanf("%d %d", &x1, &x2);
        if (read_items == EOF) {
            break;
        }
        if (read_items != 2) {
            printf("Invalid input. Example: 1 0\n");
            discard_stdin_line();
            continue;
        }
        if (x1 == -1 && x2 == -1) {
            break;
        }
        if (!is_binary_value(x1) || !is_binary_value(x2)) {
            printf("Inputs must be binary values (0 or 1).\n");
            continue;
        }

        output = predict(&p, x1, x2);
        printf("gate=%s, input=(%d,%d), predicted=%d\n", gate, x1, x2, output);

        while (1) {
            int read_response;
            printf("Is this OK? (y/n): ");
            read_response = scanf(" %c", &response);
            if (read_response == EOF) {
                goto finish;
            }
            if (read_response != 1) {
                printf("Please answer y or n.\n");
                discard_stdin_line();
                continue;
            }
            if (parse_ok_response(response, &is_ok)) {
                break;
            }
            printf("Please answer y or n.\n");
            discard_stdin_line();
        }

        target = is_ok ? output : 1 - output;
        if ( !is_ok ){
           update_weights(&p, x1, x2, target, output);
        }
        iteration++;
    }

finish:
    if (idx >= 0) {
        entries[idx].p = p;
    } else {
        if (count >= MAX_MODELS) {
            printf("Model storage full (%d gates). Could not save '%s'.\n", MAX_MODELS, gate);
            return;
        }
        strcpy(entries[count].gate, gate);
        entries[count].p = p;
        count++;
    }

    if (save_models(entries, count)) {
        printf("\nModel for gate '%s' saved to %s\n", gate, MODEL_FILE);
    } else {
        printf("\nWarning: Could not save model to %s\n", MODEL_FILE);
    }
}

/* Inference mode using the model associated with the selected gate label. */
static int run_mode(const char *gate, int x1, int x2) {
    ModelEntry entries[MAX_MODELS];
    int count;
    int idx;
    int output;
    Perceptron p;

    if (!load_models(entries, &count)) {
        printf("Could not read model storage from %s\n", MODEL_FILE);
        return 1;
    }

    idx = find_model(entries, count, gate);
    if (idx < 0) {
        printf("No saved model for gate '%s'. Train it first.\n", gate);
        return 1;
    }

    output = predict(&entries[idx].p, x1, x2);
    printf("Using gate '%s':\n", gate);
    //print_weights(&entries[idx].p);
    p=entries[idx].p;
    printf(" weights: w1=%.4f, w2=%.4f, bias=%.4f\n",p.w1, p.w2, p.bias);
    printf(" input=(%d,%d) -> output=%d\n", x1, x2, output);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        char gate[MAX_GATE_LABEL];

        if (argc != 3) {
            usage(argv[0]);
            return 1;
        }

        if (!normalize_gate_label(argv[2], gate)) {
            printf("Invalid gate label. Use letters/digits/_/- and max %d chars.\n",
                   MAX_GATE_LABEL - 1);
            return 1;
        }

        train_mode(gate);
        return 0;
    }

    if (strcmp(argv[1], "run") == 0) {
        char gate[MAX_GATE_LABEL];
        char extra;
        int x1;
        int x2;

        if (argc != 5) {
            usage(argv[0]);
            return 1;
        }

        if (!normalize_gate_label(argv[2], gate)) {
            printf("Invalid gate label. Use letters/digits/_/- and max %d chars.\n",
                   MAX_GATE_LABEL - 1);
            return 1;
        }

        if (sscanf(argv[3], "%d%c", &x1, &extra) != 1 || sscanf(argv[4], "%d%c", &x2, &extra) != 1) {
            printf("x1 and x2 must be integer values 0 or 1.\n");
            return 1;
        }

        if (!is_binary_value(x1) || !is_binary_value(x2)) {
            printf("x1 and x2 must be binary values (0 or 1).\n");
            return 1;
        }

        return run_mode(gate, x1, x2);
    }

    usage(argv[0]);
    return 1;
}
