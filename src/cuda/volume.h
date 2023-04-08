#ifndef VOLUME_H
#define VOLUME_H

#include <math.h>
#include <stdio.h>

#include "geometry.h"
#include "linalg.h"

#define N_CUBE_VERTEXES 8
#define N_DODECAHEDRON_VERTEXES 20
#define N_ICOSAHEDRON_VERTEXES 12
#define N_FLOOR_VERTEXES 4

#define N_CUBE_POLYGONS 12
#define N_DODECAHEDRON_POLYGONS 60
#define N_ICOSAHEDRON_POLYGONS 20
#define N_FLOOR_POLYGONS 2

#define N_CUBE_EDGES 12
#define N_DODECAHEDRON_EDGES 30
#define N_ICOSAHEDRON_EDGES 30

void calculate_normals(const triangle *polygons, vec3 *normals, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        normals[i] = normalize(cross(polygons[i].b - polygons[i].a, polygons[i].c - polygons[i].b));
    }
}

void calculate_lights(const vec3 *vertexes, const int *iedges, int n, vec3 center, 
                      vec3 *light_points, int m)
{
    int i, j;
    for (i = 0; i < n; i++) // цикл по рёбрам
    {
        for (j = 0; j < m; j++) // цикл по источникам света
        {
            light_points[i * m + j] = ratio_point(vertexes[iedges[i << 1]],
                                                  vertexes[iedges[(i << 1) + 1]],
                                                  (j + 1) / (float)(m - j));
            light_points[i * m + j] += 5e-1f * normalize(light_points[i * m + j] - center);
        }
    }
}

void triangle_fan_2_triangles(vec3 start, const vec3 *vertexes, int n, triangle *polygons)
{
    int i;
    for (i = 0; i < n - 1; i++)
    {
        polygons[i] = { start, vertexes[i], vertexes[i + 1] };
    }
}

void make_cube(vec3 center, float R, vec3 color, float kr, float kref,
               triangle *polygons, vec3 *normals, vec3 *colors, float *krs, float *krefs,
               int m)
{
    /*
    * 8 вершин
    * 12 полигонов
    * 12 рёбер
    */
    int i, j, offset,
        iedges[] = { 7, 6,
                     6, 5,
                     5, 4,
                     4, 7,
                     0, 1,
                     1, 2,
                     2, 3,
                     3, 0,
                     2, 6,
                     1, 5,
                     0, 4,
                     3, 7 };
    float a = 2 * R / sqrtf(3); // ребро
    vec3 light_point;
    vec3 vertexes[N_CUBE_VERTEXES];

    vertexes[0] = { center.x - a / 2, center.y - a / 2, center.z - a / 2 };
    vertexes[1] = { vertexes[0].x, vertexes[0].y + a, vertexes[0].z  };
    vertexes[2] = { vertexes[0].x + a, vertexes[0].y + a, vertexes[0].z };
    vertexes[3] = { vertexes[0].x + a, vertexes[0].y, vertexes[0].z };
    vertexes[4] = { vertexes[0].x, vertexes[0].y, vertexes[0].z + a };
    vertexes[5] = { vertexes[4].x, vertexes[4].y + a, vertexes[4].z };
    vertexes[6] = { vertexes[4].x + a, vertexes[4].y + a, vertexes[4].z };
    vertexes[7] = { vertexes[4].x + a, vertexes[4].y, vertexes[4].z };

    polygons[0] = { vertexes[0], vertexes[1], vertexes[2] };
    polygons[1] = { vertexes[2], vertexes[3], vertexes[0] };

    polygons[2] = { vertexes[6], vertexes[7], vertexes[3] };
    polygons[3] = { vertexes[3], vertexes[2], vertexes[6] };

    polygons[4] = { vertexes[2], vertexes[1], vertexes[5] };
    polygons[5] = { vertexes[5], vertexes[6], vertexes[2] };

    polygons[6] = { vertexes[4], vertexes[5], vertexes[1] };
    polygons[7] = { vertexes[1], vertexes[0], vertexes[4] };

    polygons[8] = { vertexes[3], vertexes[7], vertexes[4] };
    polygons[9] = { vertexes[4], vertexes[0], vertexes[3] };

    polygons[10] = { vertexes[6], vertexes[5], vertexes[4] };
    polygons[11] = { vertexes[4], vertexes[7], vertexes[6] };

    calculate_normals(polygons, normals, N_CUBE_POLYGONS);

    for (i = 0; i < N_CUBE_POLYGONS; i++)
    {
        colors[i] = color;
        krs[i] = kr;
        krefs[i] = kref;
    }

    for (i = 0; i < N_CUBE_EDGES; i++) // цикл по рёбрам
    {
        for (j = 0; j < m; j++)        // цикл по маленьким кубам
        {
            light_point = ratio_point(vertexes[iedges[i << 1]],
                                      vertexes[iedges[(i << 1) + 1]],
                                      (j + 1) / (float)(m - j));
            light_point -= 5e-1f * normalize(light_point - center); // центр маленького куба на ребре
            offset = N_CUBE_POLYGONS + i * m * N_CUBE_POLYGONS + j * N_CUBE_POLYGONS;
            make_cube(light_point, R / 15, { 1, 1, 1 }, 0, 0,
                      polygons + offset, normals  + offset, colors + offset, krs + offset, krefs + offset, 0);
        }
    }
}

void make_dodecahedron(vec3 center, float R, vec3 color, float kr, float kref,
                       triangle *polygons, vec3 *normals, vec3 *colors, float *krs, float *krefs,
                       int m)
{
    /*
    * 20 вершин
    * 60 полигонов
    * 30 рёбер
    */
    int i, j, offset;

    float phi = (1 + sqrtf(5)) / 2,                   // золотое сечение
          k = R / sqrtf(phi * phi + 1 / (phi * phi)); // коэффициент пропорциональности для масштабирования
    vec3 start, light_point;
    vec3 vertexes[N_DODECAHEDRON_VERTEXES], tmp_vert[6];
    int face_indicies[] = { 8, 7, 15, 3, 11,
                            4, 8, 11, 0, 12,
                            11, 3, 18, 19, 0,
                            15, 14, 2, 18, 3,
                            7, 17, 6, 14, 15,
                            4, 16, 17, 7, 8,
                            10, 1, 19, 18, 2,
                            9, 10, 2, 14, 6,
                            13, 1, 10, 9, 5,
                            12, 0, 19, 1, 13,
                            4, 12, 13, 5, 16,
                            5, 9, 6, 17, 16 },
        iedges[] = { 0, 19,
                     19, 18,
                     18, 3,
                     3, 11,
                     11, 0,
                     11, 8,
                     8, 7,
                     7, 15,
                     15, 3,
                     15, 14,
                     14, 2,
                     2, 18,
                     8, 4,
                     4, 12,
                     12, 0,
                     4, 16,
                     16, 17,
                     17, 7,
                     17, 6,
                     6, 14,
                     16, 5,
                     5, 13,
                     13, 12,
                     13, 1,
                     1, 19,
                     1, 10,
                     10, 2,
                     9, 10,
                     9, 6,
                     5, 9 };

    /*      куб      */
    /* верхняя грань */
    vertexes[0] = { -1, -1,  1 };
    vertexes[1] = { -1,  1,  1 };
    vertexes[2] = {  1,  1,  1 };
    vertexes[3] = {  1, -1,  1 };
    /* нижняя грань */
    vertexes[4] = { -1, -1, -1 };
    vertexes[5] = { -1,  1, -1 };
    vertexes[6] = {  1,  1, -1 };
    vertexes[7] = {  1, -1, -1 };

    /* прямоугольник в yz */
    vertexes[8]  = { 0, -phi, -1 / phi };
    vertexes[9]  = { 0,  phi, -1 / phi };
    vertexes[10] = { 0,  phi,  1 / phi };
    vertexes[11] = { 0, -phi,  1 / phi };

    /* прямоугольник в xy */
    vertexes[12] = { -phi, -1 / phi, 0 };
    vertexes[13] = { -phi,  1 / phi, 0 };
    vertexes[14] = {  phi,  1 / phi, 0 };
    vertexes[15] = {  phi, -1 / phi, 0 };

    /* прямоугольник в xz */
    vertexes[16] = { -1 / phi, 0, -phi };
    vertexes[17] = {  1 / phi, 0, -phi };
    vertexes[18] = {  1 / phi, 0,  phi };
    vertexes[19] = { -1 / phi, 0,  phi };

    /* масштабирование + сдвиг */
    for (i = 0; i < N_DODECAHEDRON_VERTEXES; i++)
    {
        vertexes[i] = vertexes[i] * k + center;
    }

    /* формирование полигонов */
    for (i = 0; i < 12; i++)
    {
        /* обход пятиугольника */
        tmp_vert[0] = vertexes[face_indicies[i * 5]];
        tmp_vert[1] = vertexes[face_indicies[i * 5 + 1]];
        tmp_vert[2] = vertexes[face_indicies[i * 5 + 2]];
        tmp_vert[3] = vertexes[face_indicies[i * 5 + 3]];
        tmp_vert[4] = vertexes[face_indicies[i * 5 + 4]];
        tmp_vert[5] = tmp_vert[0];

        start = mid_point(tmp_vert, 5);
        triangle_fan_2_triangles(start, tmp_vert, 6, polygons + i * 5);
    }

    calculate_normals(polygons, normals, N_DODECAHEDRON_POLYGONS);

    for (i = 0; i < N_DODECAHEDRON_POLYGONS; i++)
    {
        colors[i] = color;
        krs[i] = kr;
        krefs[i] = kref;
    }

    for (i = 0; i < N_DODECAHEDRON_EDGES; i++) // цикл по рёбрам
    {
        for (j = 0; j < m; j++)        // цикл по маленьким кубам
        {
            light_point = ratio_point(vertexes[iedges[i << 1]],
                                      vertexes[iedges[(i << 1) + 1]],
                                      (j + 1) / (float)(m - j));
            light_point -= 5e-1f * normalize(light_point - center); // центр маленького куба на ребре
            offset = N_DODECAHEDRON_POLYGONS + i * m * N_CUBE_POLYGONS + j * N_CUBE_POLYGONS;
            make_cube(light_point, R / 30, { 1, 1, 1 }, 0, 0,
                      polygons + offset, normals + offset, colors + offset, krs + offset, krefs + offset, 0);
        }
    }
}

void make_icosahedron(vec3 center, float R, vec3 color, float kr, float kref,
                      triangle *polygons, vec3 *normals, vec3 *colors, float *krs, float *krefs,
                      int m)
{
    /*
    * 12 вершин
    * 20 полигонов
    * 30 рёбер
    */

    int i, j, offset;
    float phi = (1 + sqrtf(5)) / 2,     // золотое сечение
           k = R / sqrtf(1 + phi * phi); // коэффициент пропорциональности дл¤ масштабирования
    int face_indicies[] = { 1, 8, 7,
                            0, 1, 7,
                            0, 7, 9,
                            3, 9, 6,
                            3, 6, 2,
                            2, 6, 8,
                            11, 2, 8,
                            1, 11, 8,
                            4, 11, 1,
                            4, 1, 0,
                            4, 0, 10,
                            5, 2, 11,
                            6, 7, 8,
                            9, 7, 6,
                            5, 3, 2,
                            10, 0, 9,
                            10, 9, 3,
                            10, 3, 5,
                            5, 11, 4,
                            4, 10, 5 },
        iedges[] = { 0, 7,
                     7, 1,
                     1, 0,
                     1, 8,
                     8, 7,
                     7, 6,
                     6, 8,
                     8, 2,
                     2, 6,
                     8, 11,
                     11, 2,
                     11, 1,
                     11, 4,
                     4, 1,
                     0, 4,
                     0, 9,
                     9, 7,
                     9, 6,
                     9, 3,
                     3, 6,
                     3, 2,
                     4, 5,
                     5, 11,
                     4, 10,
                     5, 10,
                     10, 7,
                     0, 10,
                     10, 3,
                     5, 3,
                     5, 2 };
    vec3 light_point;
    vec3 vertexes[N_ICOSAHEDRON_VERTEXES];

    /* прямоугольник в xy */
    vertexes[0] = { -1, -phi, 0 };
    vertexes[1] = {  1, -phi, 0 };
    vertexes[2] = {  1,  phi, 0 };
    vertexes[3] = { -1,  phi, 0 };

    /* прямоугольник в yz */
    vertexes[4] = { 0, -1, -phi };
    vertexes[5] = { 0,  1, -phi };
    vertexes[6] = { 0,  1,  phi };
    vertexes[7] = { 0, -1,  phi };

    /* прямоугольник в xz */
    vertexes[8]  = {  phi, 0,  1 };
    vertexes[9]  = { -phi, 0,  1 };
    vertexes[10] = { -phi, 0, -1 };
    vertexes[11] = {  phi, 0, -1 };

    /* масштабирование + сдвиг */
    for (i = 0; i < N_ICOSAHEDRON_VERTEXES; i++)
    {
        vertexes[i] = vertexes[i] * k + center;
    }

    /* формирование полигонов */
    for (int i = 0; i < N_ICOSAHEDRON_POLYGONS; i++)
    {
        polygons[i] = { vertexes[face_indicies[3 * i    ]], 
                        vertexes[face_indicies[3 * i + 1]],
                        vertexes[face_indicies[3 * i + 2]] };
    }

    calculate_normals(polygons, normals, N_ICOSAHEDRON_POLYGONS);

    for (i = 0; i < N_ICOSAHEDRON_POLYGONS; i++)
    {
        colors[i] = color;
        krs[i] = kr;
        krefs[i] = kref;
    }

    for (i = 0; i < N_ICOSAHEDRON_EDGES; i++) // цикл по рёбрам
    {
        for (j = 0; j < m; j++)        // цикл по маленьким кубам
        {
            light_point = ratio_point(vertexes[iedges[i << 1]],
                                      vertexes[iedges[(i << 1) + 1]],
                                      (j + 1) / (float)(m - j));
            light_point -= 5e-1f * normalize(light_point - center); // центр маленького куба на ребре
            offset = N_ICOSAHEDRON_POLYGONS + i * m * N_CUBE_POLYGONS + j * N_CUBE_POLYGONS;
            make_cube(light_point, R / 30, { 1, 1, 1 }, 0, 0,
                      polygons + offset, normals + offset, colors + offset, krs + offset, krefs + offset, 0);
        }
    }
}

void make_floor(vec3 a, vec3 b, vec3 c, vec3 d, vec3 color, float kr,
                triangle *polygons, vec3 *normals, vec3 *colors, float *krs, float *krefs,
                vec3 *ax, vec3 *ay, int w, int h)
{
    /*
    * 4 вершины
    * 2 полигона
    */

    int i;
    int pi[3];
    vec3 p, t,
         e1, e2, e3;
    float x1, x2, x3, 
          y1, y2, y3;
    float A[3 * 3], B[3], work[3];

    /* текстурные координаты */
    tex_triangle tex[] = { (float)w,            0, (float)w, (float)h,         0, (float)h,
                                   0,    (float)h,         0,         0, (float)w,         0 };

    polygons[0] = { c, b, a };
    polygons[1] = { a, d, c };

    calculate_normals(polygons, normals, 2);

    colors[0] = colors[1] = color;
    krs[0] = krs[1] = kr;
    krefs[0] = krefs[1] = 0;

    /* расчёт интерполяционных коэффициентов */
    for (i = 0; i < N_FLOOR_POLYGONS; i++)
    {
        e1 = polygons[i].a;
        e2 = polygons[i].b;
        e3 = polygons[i].c;
        t = cross(e3 - e1, e2 - e1);
        p = cross(e2 - e1, t);
        x1 = dot(e2 - e1, e1);
        x2 = dot(e2 - e1, e2);
        x3 = dot(e2 - e1, e3);
        y1 = dot(p, e1);
        y2 = dot(p, e2);
        y3 = dot(p, e3);

        A[0] = 1, A[1] = x1, A[2] = y1;
        A[3] = 1, A[4] = x2, A[5] = y2;
        A[6] = 1, A[7] = x3, A[8] = y3;

        LUP(A, 3, pi);

        B[0] = tex[i].a.x;
        B[1] = tex[i].b.x;
        B[2] = tex[i].c.x;

        LUP_solve(A, pi, B, 3, (float *)(ax + i), work);

        B[0] = tex[i].a.y;
        B[1] = tex[i].b.y;
        B[2] = tex[i].c.y;

        LUP_solve(A, pi, B, 3, (float *)(ay + i), work);
    }
}

#endif
