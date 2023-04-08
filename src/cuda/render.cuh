#ifndef RENDER_CUH
#define RENDER_CUH

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

#include "geometry.h"
#include "light.h"

struct node
{
    vec3 pos;
    vec3 dir;
    vec3 color;
    int k;       // номер полигона, с которым было столкновение
    float kr;   // коэффициент отражения
    float kref; // коэффициент прозрачности
    int num;     // номер в отсортированном массиве
    int left;    // левый сын
    int right;   // правый сын
};

__device__
node ray_dev(vec3 pos, vec3 dir,
             const triangle *polygons, const vec3 *normals, const vec3 *colors, const float *krs, const float *krefs, int n,
             const vec3 *light_points, const vec3 *light_colors, int m,
             const uchar4 *tex, int wtex, int htex, const vec3 *ax, const vec3 *ay, int tex_start)
{
    /*
    * pos           - точка, откуда идёт луч
    * dir           - направление луча
    * polygons      - полигоны
    * normals       - нормали
    * colors        - цвета полигонов
    * krs           - коэффциенты отражения
    * krefs         - коэффициенты прозрачности
    * n             - число полигонов
    * light_points  - положение источников света
    * light_colors  - цвета источников света
    * m             - число источников света
    * tex           - текстура пола
    * wtex          - длина текстуры
    * htex          - высота текстуры
    * ax            - интерполяционные коэффициенты по x
    * ay            - интерполяционные коэффициенты по y
    * tex_start     - индекс начала полигонов с текстурой
    */

    bool intersect;
    int i, j,
        k, k_min = -1;
    float ts, ts_min,
           xt, yt, tx, ty;

    vec3 p, t, r,
         local_color;

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
        return { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, -1, 0, 0, -1, -1, -1 };
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
        j = (int)fmaxf(fminf(roundf(tx - 0.5f), (float)(wtex - 1)), 0);
        i = (int)fmaxf(fminf(roundf(ty - 0.5f), (float)(htex - 1)), 0);

        local_color = brightness(r, normals[k_min], bytes2color(tex[(htex - i - 1) * wtex + j]), polygons, krefs, n,
                                 light_points, light_colors, m, pos);
    }
    return { r, dir, local_color, k_min, krs[k_min], krefs[k_min], -1, -1, -1 };
}

__global__
void zero_render_kernel(vec3 pc, vec3 pv,
                        int w, int h, float FOV,
                        const triangle *polygons, const vec3 *normals, const vec3 *colors, const float *krs, const float *krefs, int n,
                        const vec3 *light_points, const vec3 *light_colors, int m,
                        const uchar4 *tex, int wtex, int htex, const vec3 *ax, const vec3 *ay, int tex_start,
                        node *tree)
{
    /*
    * Первичный запуск
    * pc            - расположение камеры
    * pv            - точка, на которую смотрит камера
    * w             - ширина кадра
    * h             - высота кадра
    * FOV           - угол обзора
    * polygons      - полигоны
    * normals       - нормали
    * colors        - цвета полигонов
    * krs           - коэффиценты отражения
    * krefs         - коэффициенты преломления
    * n             - число полигонов
    * light_points  - положение источников света
    * light_colors  - цвета источников света
    * m             - число источников света
    * tex           - текстура пола
    * wtex          - длина текстуры
    * htex          - высота текстуры
    * ax            - интерполяционные коэффициенты по x
    * ay            - интерполяционные коэффициенты по y
    * tex_start     - индекс начала полигонов с текстурой
    * tree          - дерево для записи
    */

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y,
        offsetx = blockDim.x * gridDim.x,
        offsety = blockDim.y * gridDim.y,
        i, j;
    float dw = 2.f / (w - 1),
          dh = 2.f / (h - 1),
          z = 1 / tanf(radians(FOV / 2));
    vec3 bz = normalize(pv - pc),
         bx = normalize(cross(bz, { 0, 0, 1 })),
         by = normalize(cross(bx, bz)),
         v, dir;

    for (j = idy; j < h; j += offsety)
    {
        for (i = idx; i < w; i += offsetx)
        {
            v = { -1 + dw * (float)i, (-1 + dh * (float)j) * h / (float)w, z };
            dir = mult(bx, by, bz, v); // направление лучей

            tree[j * w + i] = ray_dev(pc, normalize(dir),
                                      polygons, normals, colors, krs, krefs, n,
                                      light_points, light_colors, m,
                                      tex, wtex, htex, ax, ay, tex_start);
            tree[j * w + i].num = j * w + i;
        }
    }
}

__global__
void render_kernel(int nrays,
                   const triangle *polygons, const vec3 *normals, const vec3 *colors, const float *krs, const float *krefs, int n,
                   const vec3 *light_points, const vec3 *light_colors, int m,
                   const uchar4 *tex, int wtex, int htex, const vec3 *ax, const vec3 *ay, int tex_start,
                   node *prev_tree, node *cur_tree)
{
    /*
    * pc            - расположение камеры
    * pv            - точка, на которую смотрит камера
    * w             - ширина кадра
    * h             - высота кадра
    * FOV           - угол обзора
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
    * prev_tree     - предыдущий уровень рекурсии
    * prev_tree     - текущий уровень рекурсии
    */

    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        offsetx = blockDim.x * gridDim.x,
        iray,
        k;
    vec3 pos, dir,
         reflect_dir, r, rr;
    float kr, kref;

    node reflected_node, refracted_node;

    for (iray = idx; iray < nrays; iray += offsetx) // цикл по лучам
    {
        
        pos =         prev_tree[iray].pos;   // точка на полигоне
        dir =         prev_tree[iray].dir;   // направление влёта в полигон
        k =           prev_tree[iray].k;     // номер полигона влёта
        kr =          prev_tree[iray].kr;
        kref =        prev_tree[iray].kref;
        
        if (kr > 0) // вычисляем отражение
        {
            reflect_dir = normalize(reflect(dir, normals[k]));
            r = pos + 1e-2f * reflect_dir;    // немного смещаемся
            reflected_node = ray_dev(r, reflect_dir,
                                     polygons, normals, colors, krs, krefs, n,
                                     light_points, light_colors, m,
                                     tex, wtex, htex, ax, ay, tex_start);
            reflected_node.num = iray << 1;
        }
        else      // не вычисляем отражение
        {
            reflected_node = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, -1, 0, 0, iray << 1, -1, -1 };
        }
        
        if (kref > 0) // вычисляем преломление
        {
            rr = pos + 1e-2f * dir;          // немного смещаемся
            refracted_node = ray_dev(rr, dir,
                                     polygons, normals, colors, krs, krefs, n,
                                     light_points, light_colors, m,
                                     tex, wtex, htex, ax, ay, tex_start);
            refracted_node.num = (iray << 1) + 1;
        }
        else
        {
            refracted_node = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, -1, 0, 0, (iray << 1) + 1, -1, -1 };
        }

        cur_tree[iray << 1]       = reflected_node;
        cur_tree[(iray << 1) + 1] = refracted_node;
        prev_tree[iray].left  = iray << 1;
        prev_tree[iray].right = (iray << 1) + 1;
    }
}

__global__ void SSAA_kernel(uchar4 *dst, int w, int h, int kw, int kh,
                            const uchar4 *src)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x,
        idy = blockIdx.y * blockDim.y + threadIdx.y,
        offsetx = blockDim.x * gridDim.x,
        offsety = blockDim.y * gridDim.y,
        x, y, i, j, index;
    float4 mid;

    for (y = idy; y < h; y += offsety)
    {
        for (x = idx; x < w; x += offsetx)
        {
            mid = { 0, 0, 0, 0 };
            for (j = 0; j < kh; j++)
            {
                for (i = 0; i < kw; i++)
                {
                    index = y * w * kw * kh + x * kw + j * w * kw + i;
                    mid.x += src[index].x;
                    mid.y += src[index].y;
                    mid.z += src[index].z;
                    mid.w += src[index].w;
                }
            }
            dst[y * w + x].x = (uchar)roundf(mid.x / (float)(kw * kh));
            dst[y * w + x].y = (uchar)roundf(mid.y / (float)(kw * kh));
            dst[y * w + x].z = (uchar)roundf(mid.z / (float)(kw * kh));
            dst[y * w + x].w = (uchar)roundf(mid.w / (float)(kw * kh));
        }
    }
}

#endif
