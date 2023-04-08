#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "cuda.h"
#include "cuda_runtime.h"

#include <vector_types.h>
#include <math.h>

typedef unsigned char uchar;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#undef M_PI
#define M_PI 3.14159265358979323846f

struct vec3
{
    float x;
    float y;
    float z;
};

struct vec2
{
    float x;
    float y;
};

struct triangle
{
    vec3 a;
    vec3 b;
    vec3 c;
};

struct tex_triangle
{
    vec2 a;
    vec2 b;
    vec2 c;
};

__host__ __device__
float dot(vec3 a, vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
vec3 cross(vec3 a, vec3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

__host__ __device__
float norm(vec3 v)
{
    return sqrtf(dot(v, v));
}

__host__ __device__
vec3 normalize(vec3 v)
{
    float l = norm(v);
    return { v.x / l, v.y / l, v.z / l };
}

__host__ __device__
vec3 mult(vec3 a, vec3 b, vec3 c,
          vec3 v)
{
    return { a.x * v.x + b.x * v.y + c.x * v.z,
             a.y * v.x + b.y * v.y + c.y * v.z,
             a.z * v.x + b.z * v.y + c.z * v.z };
}

__host__ __device__
vec3 operator +(vec3 a, vec3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__
vec3& operator +=(vec3 &a, const vec3 &b)
{
    a = a + b;
    return a;
}

__host__ __device__
vec3 operator +(vec3 a, float b)
{
    return { a.x + b, a.y + b, a.z + b };
}

__host__ __device__
vec3 operator -(vec3 a)
{
    return { -a.x, -a.y, -a.z };
}

__host__ __device__
vec3 operator -(vec3 a, vec3 b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__
vec3 operator -(vec3 a, float b)
{
    return { a.x - b, a.y - b, a.z - b };
}

__host__
vec3 &operator -=(vec3 &a, const vec3 &b)
{
    a = a - b;
    return a;
}

__host__
vec3 &operator -=(vec3 &a, float b)
{
    a = a - b;
    return a;
}

__host__ __device__
vec3 operator *(vec3 a, vec3 b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__host__ __device__
vec3 operator *(vec3 a, float b)
{
    return { a.x * b, a.y * b, a.z * b };
}

__host__ __device__
vec3 operator *(float b, vec3 a)
{
    return a * b;
}

__host__
vec3& operator *=(vec3 &a, float b)
{
    a = a * b;
    return a;
}

__host__ __device__
vec3 operator /(vec3 a, float b)
{
    return { a.x / b, a.y / b, a.z / b };
}

__host__
vec3 mid_point(const vec3 *points, int n)
{
    int i;
    vec3 mid = { 0, 0, 0 };
    for (i = 0; i < n; i++)
    {
        mid += points[i];
    }
    return mid / n;
}

__host__ __device__
vec3 ratio_point(vec3 a, vec3 b, float lambda)
{
    return { (a.x + lambda * b.x) / (lambda + 1), 
             (a.y + lambda * b.y) / (lambda + 1), 
             (a.z + lambda * b.z) / (lambda + 1) };
}

__host__ __device__
vec3 bytes2color(uchar4 rgbcol)
{
    return { rgbcol.x / 255.f, rgbcol.y / 255.f, rgbcol.z / 255.f };
}

__host__ __device__
uchar4 color2bytes(vec3 col)
{
    return { (uchar)fminf(roundf(col.x * 255), 255.f),
             (uchar)fminf(roundf(col.y * 255), 255.f),
             (uchar)fminf(roundf(col.z * 255), 255.f),
             255 };
}

__host__ __device__
float degrees(float radians)
{
    return radians * 57.295779513082320876798154814105f;
}

__host__ __device__
float radians(float degrees)
{
    return degrees * 0.01745329251994329576923690768489f;
}

__host__ __device__
vec3 threshold(vec3 v, float l, float u)
{
    return { fmaxf(fminf(v.x, u), l), fmaxf(fminf(v.y, u), l), fmaxf(fminf(v.z, u), l) };
}

void vec3_copy(const vec3 *src, vec3 *dst, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        dst[i] = src[i];
    }
}

void vec3_print(const vec3 &v)
{
    printf("[%f %f %f]\n", v.x, v.y, v.z);
}

void tex_triangle_print(const tex_triangle &t)
{
    printf("[(%f, %f), (%f, %lf), (%f, %f)]\n",
           t.a.x, t.a.y, t.b.x, t.b.y,t.c.x, t.c.y);
}

void triangle_print(const triangle &t)
{
    printf("[(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)]\n", 
            t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z);
}

void mat_print(const float *A, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        printf("[ ");
        for (j = 0; j < n; j++)
        {
            printf("%f ", A[i * n + j]);
        }
        printf("]\n");
    }
}

#endif 
