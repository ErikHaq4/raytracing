#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <algorithm>

void vector_swap(int n, float *x, int incx, float *y, int incy)
{
    if (n == 0 || (x == y && incx == incy))
        return;

    if (incx == 1 && incy == 1)
    {
        for (int i = 0; i < n; i++)
        {
            float tmp = x[i];
            x[i] = y[i];
            y[i] = tmp;
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            float tmp = x[i * incx];
            x[i * incx] = y[i * incy];
            y[i * incy] = tmp;
        }
    }
}

bool LUP(float *A, int n, int *pi, float eps=1e-8f)
{
    int i, j, k, k_;
    float tmp;

    for (i = 0; i < n; i++)
    {
        pi[i] = i;
    }

    for (k = 0; k < n; k++)
    {
        k_ = k;
        for (i = k + 1; i < n; i++)
        {
            if (fabsf(A[i * n + k]) > fabsf(A[k_ * n + k]))
            {
                k_ = i;
            }
        }

        if (fabsf(A[k_ * n + k]) < eps)
        {
            return false;
        }

        if (k != k_)
        {
            tmp = pi[k];
            pi[k] = pi[k_];
            pi[k_] = tmp;

            vector_swap(n, A + k * n, 1, A + k_ * n, 1);
        }

        for (i = k + 1; i < n; i++)
        {
            A[i * n + k] /= A[k * n + k];
            for (j = k + 1; j < n; j++)
            {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
    return true;
}

void LUP_solve(const float *LU, const int *pi, const float *b, int n,
               float *x, float *work)
{
    int i, j;
    float sum;
    float *y = work;

    std::fill(y, y + n, 0.f);

    for (i = 0; i < n; i++) // прямой ход
    {
        sum = 0;
        for (j = 0; j <= i - 1; j++)
            sum += LU[i * n + j] * y[j];
        y[i] = b[pi[i]] - sum;
    }

    for (i = n - 1; i >= 0; i--) // обратный ход
    {
        sum = 0;
        for (j = i + 1; j < n; j++)
            sum += LU[i * n + j] * x[j];
        x[i] = (y[i] - sum) / LU[i * n + i];
    }
}

#endif
