#ifndef LIGHT_H
#define LIGHT_H

#include "cuda.h"
#include "cuda_runtime.h"

#include <math.h>

#include "geometry.h"

__device__ __host__
bool intersection(vec3 pos, vec3 dir, triangle polygon,
                  float *t)
{
    /*
    * Пересечение луча и треугольника
    * pos     - координаты луча
    * dir     - направление луча
    * polygon - треугольник
    * t       - параметр
    */

    vec3 e1, e2,
         pvec, tvec, qvec;
    float det, inv_det, u, v;

    e1 = polygon.b - polygon.a;
    e2 = polygon.c - polygon.a;

    // Вычисление вектора нормали к плоскости
    pvec = cross(dir, e2);
    det = dot(e1, pvec);

    // Луч параллелен плоскости
    if (det < 1e-8f && det > -1e-8f)
    {
        return false;
    }

    inv_det = 1 / det;
    tvec = pos - polygon.a;
    u = dot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1)
    {
        return false;
    }

    qvec = cross(tvec, e1);
    v = dot(dir, qvec) * inv_det;
    if (v < 0 || u + v > 1)
    {
        return false;
    }
    *t = dot(e2, qvec) * inv_det;
    return *t > 1e-3f;
}

__device__ __host__
vec3 reflect(vec3 I, vec3 N)
{
    return I - 2 * dot(N, I) * N;
}

__device__ __host__
vec3 brightness(vec3 p, vec3 normal, vec3 p_color, const triangle *polygons, const float *krefs, int n,
                const vec3 *light_points, const vec3 *light_colors, int m,
                vec3 view_point, 
                float ka=0.2f, float kd=1.0f, float ks=0.7f, int ns=32)
{
    /*
    * p            - точка, в которой считаем освещЄнность
    * n            - нормаль к p
    * p_color      - цвет p
    * polygons     - полигоны
    * kref         - коэффициенты прозрачности
    * n            - число полигонов
    * light_point  - координаты источника света
    * light_color  - цвет источника света
    * m            - количество источников света
    * view_point   - позиция наблюдателя
    * ka           - коэффицент фоновой освещённости
    * kd           - коэффицент рассеянной освещённости
    * ks           - коэффицент зеркальной освещённости
    * ns           - степень аппроксимации зеркальной освещённости
    */
    
    bool intersect;
    int i, k;
    vec3 ambient, diffuse, specular,
         light_dir, view_dir, reflect_dir,
         light_sum = { 0, 0, 0 };
    float diff, spec, ts, min_kref;

    ambient = ka * vec3({ 1, 1, 1 }); // фоновая составляющая освещенности в точке

    for (i = 0; i < m; i++) // цикл по источникам света
    {
        light_dir = light_points[i] - p; // направление на свет

        min_kref = 1;
        for (k = 0; k < n; k++) // цикл по полигонам
        {
            intersect = intersection(p + 1e-2f * normalize(light_dir), normalize(light_dir), polygons[k], &ts);
            if (intersect && krefs[k] < min_kref) // поиск минимального коэффициента прозрачности
            {
                min_kref = krefs[k];
            }
        }

        diff = fmaxf(dot(normal, light_dir), (float)0) / (norm(normal) * norm(light_dir));
        diffuse = kd * min_kref * diff * light_colors[i];      // рассеянная составляющая освещенности в точке

        view_dir = normalize(view_point - p);
        reflect_dir = -reflect(light_dir, normal);
        spec = powf(fmaxf(dot(reflect_dir, view_dir) / norm(reflect_dir), (float)0), ns);
        specular = ks * min_kref * spec * light_colors[i];    // зеркальная составляющая освещённости в точке

        light_sum = light_sum + diffuse + specular;
    }
    return threshold((ambient + light_sum) * p_color, 0, 1);
}

#endif
