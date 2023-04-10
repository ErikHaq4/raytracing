#ifndef RENDER_H
#define RENDER_H

#include <vector_types.h>
#include <math.h>
#include <omp.h>

#include "geometry.h"
#include "light.h"

vec3 ray(const vec3 &pos, const vec3 &dir, int R, int MAX_R,
         const triangle *polygons, const vec3 *normals, const vec3 *colors, const float *krs, const float *krefs, int n,
         const vec3 *light_points, const vec3 *light_colors, int m,
         const uchar4 *tex, int w, int h, const vec3 *ax, const vec3 *ay, int tex_start)
{
    /*
    * pos           - точка, откуда идёт луч
    * dir           - направление луча
    * R             - текущий уровень рекурсии
    * MAX_R         - максимальный уровень рекурсии
    * polygons      - полигоны
    * normals       - нормали
    * colors        - цвета полигонов
    * krs           - коэффициенты отражения
    * krefs         - коэффициенты прозрачности
    * n             - число полигонов
    * light_points  - положение источников света
    * light_colors  - цвета источников света
    * m             - число источников света
    * tex           - текстура пола
    * w             - длина текстуры
    * h             - высота текстуры
    * ax            - интерполяционные коэффициенты по x
    * ay            - интерполяционные коэффициенты по y
    * tex_start     - индекс начала полигонов с текстурой
    */

    bool intersect;
    int i, j,
        k, k_min = -1;
    float ts, ts_min,
           xt, yt, tx, ty;

    vec3 p, t, r, rr, rref,
         local_color, reflected_color, refracted_color, reflect_dir;

    for (k = 0; k < n; k++) // цикл по полигонам
    {
        intersect = intersection(pos, dir, polygons[k], &ts);
        if (intersect && (k_min == -1 || ts < ts_min))
        {
            k_min = k;
            ts_min = ts;
        }
    }
    if (k_min == -1) // нет пересечений
    {
        return { 0, 0, 0 };
    }
    r = pos + dir * ts_min; // точка пересечения луча с треугольником
    if (k_min < tex_start)
    {
        local_color = brightness(r, normals[k_min], colors[k_min], polygons, krefs, n,
                                 light_points, light_colors, m, pos);
    }
    else // пол
    {
        t = cross(polygons[k_min].c - polygons[k_min].a, polygons[k_min].b - polygons[k_min].a);
        p = cross(polygons[k_min].b - polygons[k_min].a, t);
        xt = dot(polygons[k_min].b - polygons[k_min].a, r);
        yt = dot(p, r);

        tx = ax[k_min - tex_start].x + ax[k_min - tex_start].y * xt + ax[k_min - tex_start].z * yt;
        ty = ay[k_min - tex_start].x + ay[k_min - tex_start].y * xt + ay[k_min - tex_start].z * yt;
        j = (int)fmaxf(fminf(roundf(tx - 0.5f), (float)(w - 1)), 0);
        i = (int)fmaxf(fminf(roundf(ty - 0.5f), (float)(h - 1)), 0);

        local_color = brightness(r, normals[k_min], bytes2color(tex[(h - i - 1) * w + j]), polygons, krefs, n,
                                 light_points, light_colors, m, pos);
    }

    if (R >= MAX_R || krs[k_min] <= 0 && krefs[k_min] <= 0) // конец рекурсии
    {
        return local_color;
    }

    reflected_color = refracted_color = { 0, 0, 0 };

    if(krs[k_min] > 0) // просчёт отражённого луча
    {
        reflect_dir = normalize(reflect(dir, normals[k_min]));

        rr = r + 1e-2f * reflect_dir; // немного смещаемся

        reflected_color = ray(rr, reflect_dir, R + 1, MAX_R,
                              polygons, normals, colors, krs, krefs, n,
                              light_points, light_colors, m,
                              tex, w, h, ax, ay, tex_start);
    }
    if (krefs[k_min] > 0) // просчёт преломлённого луча
    {
        rref = r + 1e-2f * dir;     // немного смещаемся

        refracted_color = ray(rref, dir, R + 1, MAX_R,
                              polygons, normals, colors, krs, krefs, n,
                              light_points, light_colors, m,
                              tex, w, h, ax, ay, tex_start);
    }
    return (1 - krs[k_min] - krefs[k_min]) * local_color +
           krs[k_min] * reflected_color + 
           krefs[k_min] * refracted_color;
}

void render(const vec3 &pc, const vec3 &pv,
            int w, int h, float FOV, int MAX_R,
            const triangle *polygons, const vec3 *normals, const vec3 *colors, const float *krs, const float *krefs, int n,
            const vec3 *light_points, const vec3 *light_colors, int m,
            uchar4 *pixels,
            const uchar4 *tex, int wtex, int htex, const vec3 *ax, const vec3 *ay, int tex_start,
            int threads=1)
{
    /*
    * pc            - расположение камеры
    * pv            - точка, на которую смотрит камера
    * w             - ширина кадра
    * h             - высота кадра
    * FOV           - угол обзора
    * R_MAX         - максимальная глубина рекурсии
    * polygons      - полигоны
    * normals       - нормали
    * colors        - цвета полигонов
    * krs           - коэффиценты отражения
    * krefs         - коэффициенты преломления
    * n             - число полигонов
    * light_points  - положение источников света
    * light_colors  - цвета источников света
    * m             - число источников света
    * pixels        - входное изображение
    * tex           - текстура пола
    * w             - длина текстуры
    * h             - высота текстуры
    * ax            - интерполяционные коэффициенты по x
    * ay            - интерполяционные коэффициенты по y
    * tex_start     - индекс начала полигонов с текстурой
    * threads       - число потоков
    */

    int i, j;
    float dw = 2.f / (w - 1),
          dh = 2.f / (h - 1),
           z = 1 / tanf(radians(FOV / 2)); // глубина, с которой выпускаем лучи
    vec3 bz = normalize(pv - pc),
         bx = normalize(cross(bz, { 0, 0, 1 })),
         by = normalize(cross(bx, bz)),
         v, dir;

    int size = h;
    int block_size = (size - 1) / threads + 1;

#pragma omp parallel num_threads(threads) private(i, j, v, dir)
    {
        int id = omp_get_thread_num(),
            nthrs = omp_get_num_threads();

        int begin = id * block_size,
            end = MIN((id + 1) * block_size, size);

        for (j = begin; j < end; j++)
        {
            for (i = 0; i < w; i++)
            {
                v = { -1 + dw * (float)i, (-1 + dh * (float)j) * h / (float)w, z };
                dir = mult(bx, by, bz, v); // направление лучей
                pixels[(h - 1 - j) * w + i] = color2bytes(ray(pc, normalize(dir), 0, MAX_R,
                                                              polygons, normals, colors, krs, krefs, n,
                                                              light_points, light_colors, m,
                                                              tex,
                                                              wtex, htex, ax, ay, tex_start));
            }
        }
    }
}

void SSAA(uchar4 *dst, int w, int h, int kw, int kh,
          const uchar4 *src, int threads=1)
{
    int x, y, i, j, index;
    float4 mid;

    int size = h;
    int block_size = (size - 1) / threads + 1;

#pragma omp parallel num_threads(threads) private(y, x, j, i, mid, index)
    {
        int id = omp_get_thread_num(),
            nthrs = omp_get_num_threads();

        int begin = id * block_size,
            end = MIN((id + 1) * block_size, size);

        for (y = begin; y < end; y++)
        {
            for (x = 0; x < w; x++)
            {
                mid = { 0, 0, 0, 0 };
                for (j = 0; j < kh; j++)
                {
                    for (i = 0; i < kw; i++)
                    {
                        index = y * w * kw * kh + x * kw + j * w * kw + i;
                        mid.x += (float)src[index].x;
                        mid.y += (float)src[index].y;
                        mid.z += (float)src[index].z;
                        mid.w += (float)src[index].w;
                    }
                }
                dst[y * w + x].x = (uchar)roundf(mid.x / (float)(kw * kh));
                dst[y * w + x].y = (uchar)roundf(mid.y / (float)(kw * kh));
                dst[y * w + x].z = (uchar)roundf(mid.z / (float)(kw * kh));
                dst[y * w + x].w = (uchar)roundf(mid.w / (float)(kw * kh));
            }
        }
    }
}

#endif
