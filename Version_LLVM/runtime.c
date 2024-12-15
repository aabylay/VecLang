#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double* data;
    int length;
} Vector;

typedef struct {
    double* data;
    int rows;
    int cols;
} Matrix;

/** Result structure for knn_bf */
typedef struct {
    int* knns;       // array of indices
    double* distances; // array of distances
    int k;
} KNNResult;

/** Result structure for knn_bf_batch */
typedef struct {
    double min_sum;
    int* closest_indices;   // of length m1_rows
    double* closest_distances; // of length m1_rows
    int m1_rows;
} KNBBatchResult;

/** Creates a new Vector */
Vector* vector_create(double* data, int length) {
    Vector* v = malloc(sizeof(Vector));
    if (!v) {
        fprintf(stderr, "Failed to allocate Vector\n");
        exit(1);
    }
    v->length = length;
    v->data = malloc(sizeof(double)*length);
    if (!v->data) {
        fprintf(stderr, "Failed to allocate Vector data\n");
        exit(1);
    }
    for (int i = 0; i < length; i++) {
        v->data[i] = data[i];
    }
    return v;
}

/** Creates a new Matrix */
Matrix* matrix_create(double* data, int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (!m) {
        fprintf(stderr, "Failed to allocate Matrix\n");
        exit(1);
    }
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(sizeof(double)*rows*cols);
    if (!m->data) {
        fprintf(stderr, "Failed to allocate Matrix data\n");
        exit(1);
    }
    for (int i = 0; i < rows*cols; i++) {
        m->data[i] = data[i];
    }
    return m;
}

/** Adds two vectors element-wise */
Vector* vector_add(Vector* v1, Vector* v2) {
    if (v1->length != v2->length) {
        fprintf(stderr, "vector_add: length mismatch\n");
        exit(1);
    }
    Vector* out = malloc(sizeof(Vector));
    out->length = v1->length;
    out->data = malloc(sizeof(double)*out->length);
    for (int i = 0; i < out->length; i++) {
        out->data[i] = v1->data[i] + v2->data[i];
    }
    return out;
}

/** Dot product of two vectors */
double vector_dot(Vector* v1, Vector* v2) {
    if (v1->length != v2->length) {
        fprintf(stderr, "vector_dot: length mismatch\n");
        exit(1);
    }
    double sum = 0.0;
    for (int i = 0; i < v1->length; i++) {
        sum += v1->data[i]*v2->data[i];
    }
    return sum;
}

/** Scalar multiplication of vector */
Vector* vector_scalar_mul(Vector* v, double scalar) {
    Vector* out = malloc(sizeof(Vector));
    out->length = v->length;
    out->data = malloc(sizeof(double)*out->length);
    for (int i = 0; i < v->length; i++) {
        out->data[i] = v->data[i] * scalar;
    }
    return out;
}

/**
 * similarity:
 * method = 0 => cosine similarity
 * method = 1 => euclidean distance
 */
double similarity(Vector* v1, Vector* v2, int method) {
    if (v1->length != v2->length) {
        fprintf(stderr, "similarity: length mismatch\n");
        exit(1);
    }

    if (method == 0) {
        // cosine similarity
        double dot = 0.0;
        double norm1 = 0.0, norm2 = 0.0;
        for (int i = 0; i < v1->length; i++) {
            double x = v1->data[i];
            double y = v2->data[i];
            dot += x*y;
            norm1 += x*x;
            norm2 += y*y;
        }
        double denom = sqrt(norm1)*sqrt(norm2);
        return denom == 0.0 ? 0.0 : dot/denom;
    } else {
        // euclidean distance
        double sum = 0.0;
        for (int i = 0; i < v1->length; i++) {
            double diff = v1->data[i] - v2->data[i];
            sum += diff*diff;
        }
        return sqrt(sum);
    }
}

/** Compute distance-like measure from similarity calls:
 *  For cosine: distance = 1 - similarity
 *  For euclidean: distance = euclidean distance directly
 */
static double compute_distance(Vector* v1, Vector* v2, int method) {
    if (method == 0) {
        double sim = similarity(v1, v2, 0);
        return 1.0 - sim;
    } else {
        // method == 1 (euclidean)
        return similarity(v1, v2, 1); // returns distance
    }
}

/** Extract a row from matrix M as a Vector. Caller must free the vector after use. */
static Vector* matrix_get_row(Matrix* M, int row) {
    if (row < 0 || row >= M->rows) {
        fprintf(stderr, "matrix_get_row: index out of range\n");
        exit(1);
    }
    Vector* v = malloc(sizeof(Vector));
    v->length = M->cols;
    v->data = malloc(sizeof(double)*M->cols);
    for (int i = 0; i < M->cols; i++) {
        v->data[i] = M->data[row*M->cols + i];
    }
    return v;
}

/** knn_bf:
 * Given a vector v and a matrix M, find k-nearest neighbors of v in M.
 * method=0 (cosine), method=1 (euclidean)
 * Returns a KNNResult* (which you must free after use)
 */
KNNResult* knn_bf(Vector* v, Matrix* M, int k, int method) {
    if (k > M->rows) {
        fprintf(stderr, "knn_bf: k greater than number of rows\n");
        exit(1);
    }

    // Compute distances
    double* dist_array = malloc(sizeof(double)*M->rows);
    int* indices = malloc(sizeof(int)*M->rows);

    for (int i = 0; i < M->rows; i++) {
        Vector* row_vec = matrix_get_row(M, i);
        double d = compute_distance(v, row_vec, method);
        // free row_vec
        free(row_vec->data);
        free(row_vec);
        dist_array[i] = d;
        indices[i] = i;
    }

    // Sort by distance ascending (simple bubble sort for demo)
    for (int i = 0; i < M->rows-1; i++) {
        for (int j = 0; j < M->rows-1-i; j++) {
            if (dist_array[j] > dist_array[j+1]) {
                double tmpd = dist_array[j];
                dist_array[j] = dist_array[j+1];
                dist_array[j+1] = tmpd;

                int tmpi = indices[j];
                indices[j] = indices[j+1];
                indices[j+1] = tmpi;
            }
        }
    }

    // Take top k
    KNNResult* result = malloc(sizeof(KNNResult));
    result->k = k;
    result->knns = malloc(sizeof(int)*k);
    result->distances = malloc(sizeof(double)*k);
    for (int i = 0; i < k; i++) {
        result->knns[i] = indices[i];
        result->distances[i] = dist_array[i];
    }

    free(dist_array);
    free(indices);
    return result;
}

/** knn_bf_batch:
 * For each vector in M1, find closest vector in M2.
 * method=0 (cosine), method=1 (euclidean)
 * Returns a KNBBatchResult*
 */
KNBBatchResult* knn_bf_batch(Matrix* M1, Matrix* M2, int method) {
    // For each row in M1, find closest in M2
    int m1_rows = M1->rows;
    int* closest_indices = malloc(sizeof(int)*m1_rows);
    double* closest_dists = malloc(sizeof(double)*m1_rows);

    double sum_dist = 0.0;

    for (int i = 0; i < m1_rows; i++) {
        Vector* v = matrix_get_row(M1, i);
        double best_dist = -1.0;
        int best_index = -1;
        for (int j = 0; j < M2->rows; j++) {
            Vector* w = matrix_get_row(M2, j);
            double d = compute_distance(v, w, method);
            free(w->data);
            free(w);
            if (best_dist < 0 || d < best_dist) {
                best_dist = d;
                best_index = j;
            }
        }
        free(v->data);
        free(v);
        closest_indices[i] = best_index;
        closest_dists[i] = best_dist;
        sum_dist += best_dist;
    }

    KNBBatchResult* res = malloc(sizeof(KNBBatchResult));
    res->min_sum = sum_dist;
    res->closest_indices = closest_indices;
    res->closest_distances = closest_dists;
    res->m1_rows = m1_rows;
    return res;
}
