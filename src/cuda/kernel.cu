#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <set>
#include <vector>
#include <string>
#include <ctime>
#include <ratio>
#include <chrono>

#include "render.h"
#include "render.cuh"
#include "geometry.h"
#include "volume.h"

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::string;
using std::set;
using std::vector;
using std::pair;
using thrust::device_ptr;
using thrust::partition;
using thrust::sort;

// to do: 1) добавить выравнивание всех массивов на 64 (сейчас используется выравнивание на 32)
//        2) использование текстурной памяти
//        3) придумать более эффективный алгоритм преобразования рекурсии в итерацию на GPU

//#define DEBUG

// CudaSafeCall

#ifdef DEBUG

#define CSC(call)                                                   \
do                                                                  \
{                                                                   \
    cudaError_t _res = call;                                        \
    if (_res != cudaSuccess)                                        \
    {                                                               \
        fprintf(stdout, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(_res));      \
		exit(0);												    \
    }                                                               \
} while(0)

#define DCSC(call) CSC(call)

#else // !DEBUG

#define CSC(call) call

#define DCSC(call)

#endif

#define CUDA_THREADS 32
#define CUDA_BLOCKS 65536

#define CUDA_THREADS2 {32, 8}
#define CUDA_BLOCKS2 {64, 128}

pair<dim3, dim3> optimal_grid(int n, int max_blocks = CUDA_BLOCKS, int threads = CUDA_THREADS)
{
    int n_blocks = 0;
    int n_threads = 0;

    if (n == 0)
    {
        return { dim3(1), dim3(32) }; // 1 варп
    }
    else if (n < threads)
    {
        n_blocks = 1;
        n_threads = (n + 31) & ~31; // ближайшее кратное 32 сверху
    }
    else
    {
        n_blocks = MIN((n - 1) / threads + 1, max_blocks);
        n_threads = threads;
    }

    return { dim3(n_blocks), dim3(n_threads) };
}

pair<dim3, dim3> optimal_grid2(int m, int n, pair<int, int> max_blocks = CUDA_BLOCKS2, pair<int, int> threads = CUDA_THREADS2)
{
    if (m == 0 || n == 0)
    {
        return { dim3(1, 1), dim3(32, 1) }; // 1 варп
    }

    dim3 n_blocks;
    dim3 n_threads;

    n_blocks.z = 1;
    n_threads.z = 1;

    if (n < threads.first)
    {
        n_blocks.x = 1;
        n_threads.x = (n + 31) & ~31; // ближайшее кратное 32 сверху
    }
    else
    {
        n_blocks.x = MIN((n - 1) / threads.first + 1, max_blocks.first);
        n_threads.x = threads.first;
    }

    if (m < threads.second)
    {
        n_blocks.y = 1;
        n_threads.y = m;
    }
    else
    {
        n_blocks.y = MIN((m - 1) / threads.second + 1, max_blocks.second);
        n_threads.y = threads.second;
    }

    return { n_blocks, n_threads };
}

vec3 calculate_color(int num, int R, int MAX_R,
                     node **tree)
{
    node nod = tree[R][num];
    vec3 local_color = nod.color,
         reflected_color = { 0, 0, 0 },
         refracted_color = { 0, 0, 0 };

    if (R == MAX_R || (nod.kr <= 0 && nod.kref <= 0) || nod.k == -1)
    {
        return local_color;
    }
    if (nod.left != -1)
    {
        reflected_color = calculate_color(nod.left, R + 1, MAX_R,
                                          tree);
    }
    if (nod.right != -1)
    {
        refracted_color = calculate_color(nod.right, R + 1, MAX_R,
                                          tree);
    }
    return (1 - nod.kr - nod.kref) * local_color +
           nod.kr * reflected_color +
           nod.kref * refracted_color;
}

struct partion_pred
{
    __host__ __device__
    bool operator()(node n)
    {
        return n.k >= 0 && (n.kr > 0 || n.kref > 0);
    }
};

struct sort_greater
{
    __host__ __device__
        bool operator()(node a, node b)
    {
        return a.num < b.num;
    }
};

int main(int argc, char **argv)
{
    bool is_GPU = true;
    int i, x, y, k,
        w, h, ws, hs, fov,
        frames,
        k_SSAA,
        wtex, htex,
        n_lights, MAX_R, R,
        n_cube_lights, n_dodecahedron_lights, n_icosahedron_lights, 
        n_cube_polygons, n_dodecahedron_polygons, n_icosahedron_polygons, n_floor_polygons,
        n_polygons, tex_start,
        n_rays, sum_rays;
    int *n_levels, *n_levels_size;

    float t, dt,
          rc, r0c, Arc, omegarc, prc,
          zc, z0c, Azc, omegazc, pzc,
          phic, phi0c, omegaphic,
          rv, r0v, Arv, omegarv, prv,
          zv, z0v, Azv, omegazv, pzv,
          phiv, phi0v, omegaphiv,
          cube_R, cube_kr, cube_kref,
          dodecahedron_R, dodecahedron_kr, dodecahedron_kref,
          icosahedron_R, icosahedron_kr, icosahedron_kref,
          floor_kr;
    float *krs, *krefs, 
          *cube_krs, *cube_krefs, 
          *dodecahedron_krs, *dodecahedron_krefs,
          *icosahedron_krs, *icosahedron_krefs,
          *floor_krs, *floor_krefs;
    float *krs_dev, *krefs_dev;

    char buff[256], path2images[256];
    uchar4 *MEM_CPU = NULL, *pixels, *pixels_SSAA = NULL, *tex;
    uchar4 *MEM_GPU = NULL, *pixels_dev, *pixels_SSAA_dev, *tex_dev;
    FILE *fp, *in = stdin;
    high_resolution_clock::time_point time_start, time_end;
    milliseconds time_span;
    set<string> flags; // обработка ключей запуска
    cudaError_t res;
    device_ptr<node> mid, ptr;

    vec3 pc, pv,
         cube_center, cube_color,
         dodecahedron_center, dodecahedron_color,
         icosahedron_center, icosahedron_color,
         floor_a, floor_b, floor_c, floor_d, floor_color;

    vec3 ax[N_FLOOR_POLYGONS], ay[N_FLOOR_POLYGONS];

    vec3 *normals, *colors, 
         *light_points, *light_colors,
         *cube_normals, *cube_colors,
         *dodecahedron_normals, *dodecahedron_colors,
         *icosahedron_normals, *icosahedron_colors,
         *floor_normals, *floor_colors;
    vec3 *normals_dev, *colors_dev,
         *light_points_dev, *light_colors_dev,
         *ax_dev, *ay_dev;

    triangle *polygons,
             *cube_polygons,
             *dodecahedron_polygons,
             *icosahedron_polygons,
             *floor_polygons;
    triangle *polygons_dev;

    node **tree = NULL, **tree_dev = NULL;

    /*int blocks = CUDA_BLOCKS;
    pair<int, int> blocks2 = CUDA_BLOCKS2;*/
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blocks = deviceProp.maxGridSize[0];
    pair<int, int> blocks2 = { blocks, deviceProp.maxGridSize[1] };

    int threads = CUDA_THREADS;
    pair<int, int> threads2 = CUDA_THREADS2;

    //int omp_threads = 8;
    int omp_threads = omp_get_max_threads();

    omp_threads = MIN(omp_threads, omp_get_max_threads());
    omp_set_num_threads(omp_threads);

    /* Ключи запуска */
    for (i = 1; i < argc; i++)
    {
        flags.insert(argv[i]);
    }

    if (flags.find("--cpu") != flags.end())
    {
        is_GPU = false;
    }
    if (flags.find("--gpu") != flags.end())
    {
        is_GPU = true;
    }
    if (flags.find("--default") != flags.end())
    {
        in = fopen("in.txt", "r");
        if (!in)
        {
            printf("Error: can't open %s in order to default run\n", "in.txt");
            return 0;
        }
        printf("read params from %s:\n", "in.txt");
    }

    fscanf(in, "%d", &frames);              // количество кадров
    dt = 2 * M_PI / (float)(frames - 1);    // шаг по времени
    fscanf(in, "%s", path2images);          // путь к выходным изображениям
    fscanf(in, "%d %d %d", &w, &h, &fov);   // ширина, высота, угол обзора

    /* Параметры движения камеры */
    fscanf(in, "%f %f %f", &r0c, &z0c, &phi0c);
    fscanf(in, "%f %f", &Arc, &Azc);
    fscanf(in, "%f %f %f", &omegarc, &omegazc, &omegaphic);
    fscanf(in, "%f %f", &prc, &pzc);

    /* Параметры направления камеры */
    fscanf(in, "%f %f %f", &r0v, &z0v, &phi0v);
    fscanf(in, "%f %f", &Arv, &Azv);
    fscanf(in, "%f %f %f", &omegarv, &omegazv, &omegaphiv);
    fscanf(in, "%f %f", &prv, &pzv);

    /* Параметры куба */
    fscanf(in, "%f %f %f %f %f %f %f %f %f %d",
           &cube_center.x, &cube_center.y, &cube_center.z, 
           &cube_R,
           &cube_color.x, &cube_color.y, &cube_color.z,
           &cube_kr, &cube_kref,
           &n_cube_lights);
    
    /* Параметры додекаэдра */
    fscanf(in, "%f %f %f %f %f %f %f %f %f %d",
           &dodecahedron_center.x, &dodecahedron_center.y, &dodecahedron_center.z,
           &dodecahedron_R,
           &dodecahedron_color.x, &dodecahedron_color.y, &dodecahedron_color.z,
           &dodecahedron_kr, &dodecahedron_kref,
           &n_dodecahedron_lights);

    /* Параметры икосаэдра */
    fscanf(in, "%f %f %f %f %f %f %f %f %f %d",
           &icosahedron_center.x, &icosahedron_center.y, &icosahedron_center.z,
           &icosahedron_R,
           &icosahedron_color.x, &icosahedron_color.y, &icosahedron_color.z,
           &icosahedron_kr, &icosahedron_kref,
           &n_icosahedron_lights);
    
    /* Параметры пола */
    fscanf(in, "%f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %f",
           &floor_a.x, &floor_a.y, &floor_a.z, 
           &floor_b.x, &floor_b.y, &floor_b.z,
           &floor_c.x, &floor_c.y, &floor_c.z,
           &floor_d.x, &floor_d.y, &floor_d.z,
           buff,
           &floor_color.x, &floor_color.y, &floor_color.z,
           &floor_kr);

    fp = fopen(buff, "rb");
    if (!fp)
    {
        printf("Error: can't open %s\n", buff);
        return 0;
    }
    fread(&wtex, sizeof(int), 1, fp);
    fread(&htex, sizeof(int), 1, fp);

    fscanf(in, "%d", &n_lights);

    n_cube_polygons = N_CUBE_POLYGONS + N_CUBE_EDGES * n_cube_lights * N_CUBE_POLYGONS;
    n_dodecahedron_polygons = N_DODECAHEDRON_POLYGONS + N_DODECAHEDRON_EDGES * n_dodecahedron_lights * N_CUBE_POLYGONS;
    n_icosahedron_polygons = N_ICOSAHEDRON_POLYGONS + N_ICOSAHEDRON_EDGES * n_icosahedron_lights * N_CUBE_POLYGONS;
    n_floor_polygons = N_FLOOR_POLYGONS;

    n_polygons = n_cube_polygons + n_dodecahedron_polygons + n_icosahedron_polygons + n_floor_polygons;

    tex_start = n_polygons - n_floor_polygons; // индекс начала текстурных полигонов

    /* ПЕРВОЕ ВЫДЕЛЕНИЕ ПАМЯТИ НА CPU */
    MEM_CPU = (uchar4 *)malloc(w * h * sizeof(uchar4) +                       // экран
                               wtex * htex * sizeof(uchar4)   +               // текстура
                               n_polygons             * sizeof(triangle) +    // полигоны
                               2 * n_polygons         * sizeof(vec3) +        // нормали + цвета
                               2 * n_lights           * sizeof(vec3)   +      // глобальные источники света
                               2 * n_polygons         * sizeof(float));       // коэфф. отражения + коэфф. прозрачности
    
    if (!MEM_CPU)
    {
        printf("Error: Not enough CPU memory to make operation\n");
        return 0;
    }
    pixels = MEM_CPU;
    tex = pixels + w * h;

    polygons = (triangle *)(tex + wtex * htex);
    cube_polygons = polygons;
    dodecahedron_polygons = cube_polygons + n_cube_polygons;
    icosahedron_polygons = dodecahedron_polygons + n_dodecahedron_polygons;
    floor_polygons = icosahedron_polygons + n_icosahedron_polygons;

    normals = (vec3 *)(floor_polygons + n_floor_polygons);
    cube_normals = normals;
    dodecahedron_normals = cube_normals + n_cube_polygons;
    icosahedron_normals = dodecahedron_normals + n_dodecahedron_polygons;
    floor_normals = icosahedron_normals + n_icosahedron_polygons;

    colors = floor_normals + n_floor_polygons;
    cube_colors = colors;
    dodecahedron_colors = cube_colors + n_cube_polygons;
    icosahedron_colors = dodecahedron_colors + n_dodecahedron_polygons;
    floor_colors = icosahedron_colors + n_icosahedron_polygons;

    krs = (float *)(floor_colors + n_floor_polygons);
    cube_krs = krs;
    dodecahedron_krs = cube_krs + n_cube_polygons;
    icosahedron_krs = dodecahedron_krs + n_dodecahedron_polygons;
    floor_krs = icosahedron_krs + n_icosahedron_polygons;

    krefs = floor_krs + n_floor_polygons;
    cube_krefs = krefs;
    dodecahedron_krefs = cube_krefs + n_cube_polygons;
    icosahedron_krefs = dodecahedron_krefs + n_dodecahedron_polygons;
    floor_krefs = icosahedron_krefs + n_icosahedron_polygons;

    light_points = (vec3 *)(floor_krefs + n_floor_polygons);
    light_colors = light_points + n_lights;

    /* КОНЕЦ ПЕРВОГО ВЫДЕЛЕНИЯ ПАМЯТИ НА CPU */

    fread(tex, sizeof(uchar4), wtex * htex, fp);
    fclose(fp);

    /* Параметры источников света */
    for (i = 0; i < n_lights; i++)
    {
        fscanf(in, "%f %f %f %f %f %f",
               (float *)light_points + 3 * i, (float *)light_points + 3 * i + 1, (float *)light_points + 3 * i + 2,
               (float *)light_colors + 3 * i, (float *)light_colors + 3 * i + 1, (float *)light_colors + 3 * i + 2);
    }

    /* Максимальная глубина рекурсии */
    fscanf(in, "%d", &MAX_R);

    /* SSAA */
    fscanf(in, "%d", &k_SSAA);

    ws = w * k_SSAA;
    hs = h * k_SSAA;

    /* Выделение памяти под SSAA для CPU */
    pixels_SSAA = (uchar4 *)malloc(ws * hs * sizeof(uchar4));
    if (!pixels_SSAA)
    {
        printf("Error: Not enough CPU memory to make operation\n");
        goto FREE1;
    }

    if (in != stdin)
    {
        fclose(in);
    }
    if (flags.find("--default") != flags.end()) // выводим файл в stdout
    {
        in = fopen("in.txt", "rb");

        if (!in)
        {
            printf("Error: Can't open %s\n", "in.txt");
            goto FREE2;
        }

        int size_buf = sizeof(buff) / sizeof(char);
        int chars_read = 0;

        while ((chars_read = fread(buff, sizeof(char), size_buf - 1, in)) > 0)
        {
            buff[chars_read] = '\0';
            printf("%s", buff);
        }

        printf("=======================read success=======================\n");
        fclose(in);
    }

    make_cube(cube_center, cube_R, cube_color, cube_kr, cube_kref,
              cube_polygons, cube_normals, cube_colors, 
              cube_krs, cube_krefs, 
              n_cube_lights);

    make_dodecahedron(dodecahedron_center, dodecahedron_R, dodecahedron_color, dodecahedron_kr, dodecahedron_kref,
                      dodecahedron_polygons, dodecahedron_normals, dodecahedron_colors, 
                      dodecahedron_krs, dodecahedron_krefs,
                      n_dodecahedron_lights);

    make_icosahedron(icosahedron_center, icosahedron_R, icosahedron_color, icosahedron_kr, icosahedron_kref,
                     icosahedron_polygons, icosahedron_normals, icosahedron_colors, 
                     icosahedron_krs, icosahedron_krefs,
                     n_icosahedron_lights);

    make_floor(floor_a, floor_b, floor_c, floor_d, floor_color, floor_kr,
               floor_polygons, floor_normals, floor_colors, floor_krs, floor_krefs,
               ax, ay, wtex, htex);

    if (is_GPU)
    {
        /* Выделение памяти на GPU */
        res = cudaMalloc(&MEM_GPU, 
                         (w * h + ws * hs)     * sizeof(uchar4) +     // экран + растянутый экран
                         wtex * htex           * sizeof(uchar4) +     // текстура пола
                         n_polygons            * sizeof(triangle) +   // полигоны
                         2 * n_polygons        * sizeof(vec3) +       // нормали + цвета
                         2 * n_lights          * sizeof(vec3) +       // глобальные источники света
                         2 * n_polygons        * sizeof(float) +      // krs + krefs
                         2 * N_FLOOR_POLYGONS  * sizeof(vec3));       // ax, ay

        if (res != cudaSuccess)
        {
            printf("Error: Not enough GPU memory to make operation\n");
            goto FREE2;
        }

        pixels_dev = MEM_GPU;
        pixels_SSAA_dev = pixels_dev + w * h;
        tex_dev = pixels_SSAA_dev + ws * hs;

        polygons_dev = (triangle *)(tex_dev + wtex * htex);
        normals_dev = (vec3 *)(polygons_dev + n_polygons);
        colors_dev = normals_dev + n_polygons;

        light_points_dev = (vec3 *)(colors_dev + n_polygons);
        light_colors_dev = light_points_dev + n_lights;

        krs_dev = (float *)(light_colors_dev + n_lights);
        krefs_dev = krs_dev + n_polygons;
        
        ax_dev = (vec3 *)(krefs_dev + n_polygons);
        ay_dev = ax_dev + N_FLOOR_POLYGONS;

        CSC(cudaMemcpyAsync(tex_dev, tex, wtex * htex * sizeof(uchar4), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(polygons_dev, polygons, n_polygons * sizeof(triangle), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(normals_dev, normals, n_polygons * sizeof(vec3), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(colors_dev, colors, n_polygons * sizeof(vec3), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(light_points_dev, light_points, n_lights * sizeof(vec3), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(light_colors_dev, light_colors, n_lights * sizeof(vec3), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(krs_dev, krs, n_polygons * sizeof(float), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(krefs_dev, krefs, n_polygons * sizeof(float), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(ax_dev, ax, N_FLOOR_POLYGONS * sizeof(vec3), cudaMemcpyHostToDevice));
        CSC(cudaMemcpyAsync(ay_dev, ay, N_FLOOR_POLYGONS * sizeof(vec3), cudaMemcpyHostToDevice));
        /* Конец выделения памяти на GPU */
    }
    
    printf("%s run\n", is_GPU ? "GPU" : "CPU");
    printf("omp_threads = %d\n", omp_threads);
    if (is_GPU)
    {
        printf("using %s\n", deviceProp.name);
    }
    printf("n_polygons = %d\n", n_polygons);
    printf("%12s|%12s|%12s\n",
           "frame_number", "time2render", "num of rays");

    if (is_GPU)
    {
        /* Выделенение памяти под указатели на элементы в каждом уровне */
        tree = (node **)malloc(2 * (MAX_R + 1) * sizeof(node *) +
                               2 * (MAX_R + 1) * sizeof(int));
        tree_dev = tree + MAX_R + 1;
        n_levels = (int *)(tree_dev + MAX_R + 1);
        n_levels_size = n_levels + MAX_R + 1;

        std::fill(tree, tree + MAX_R + 1, (node *)NULL);
        std::fill(tree_dev, tree_dev + MAX_R + 1, (node *)NULL);

        tree[0] = (node *)malloc(ws * hs * sizeof(node)); // выделение памяти под нулевой уровень рекурсии
        if (!tree[0])
        {
            printf("Error: Not enough CPU memory to make operation\n");
            goto FREE;
        }
        CSC(cudaMalloc(tree_dev, ws * hs * sizeof(node)));

        std::fill(n_levels, n_levels + 2 * (MAX_R + 1), 0);
        n_levels[0] = ws * hs;
        n_levels_size[0] = ws * hs;

        auto zero_render_grid = optimal_grid2(hs, ws, blocks2, threads2);
        auto SSAA_grid = optimal_grid2(h, w, blocks2, threads2);

        for (i = 0; i < frames; i++)
        {
            time_start = high_resolution_clock::now();

            t = (float)i * dt;

            rc = r0c + Arc * sinf(omegarc * t + prc);
            zc = z0c + Azc * sinf(omegazc * t + pzc);
            phic = phi0c + omegaphic * t;

            rv = r0v + Arv * sinf(omegarv * t + prv);
            zv = z0v + Azv * sinf(omegazv * t + pzv);
            phiv = phi0v + omegaphiv * t;

            pc = { rc * cosf(phic), rc * sinf(phic), zc };
            pv = { rv * cosf(phiv), rv * sinf(phiv), zv };

            sum_rays = ws * hs;

            zero_render_kernel<<<zero_render_grid.first, zero_render_grid.second>>>
                                (pc, pv, ws, hs, fov,
                                 polygons_dev, normals_dev, colors_dev, krs_dev, krefs_dev, n_polygons,
                                 light_points_dev, light_colors_dev, n_lights,
                                 tex_dev, wtex, htex, ax_dev, ay_dev, tex_start,
                                 tree_dev[0]);
            DCSC(cudaDeviceSynchronize());
            DCSC(cudaGetLastError());

            for (R = 1; R <= MAX_R; R++) // цикл по уровням
            {
                ptr = thrust::device_pointer_cast(tree_dev[R - 1]);
                mid = partition(thrust::device, 
                                ptr, ptr + n_levels[R - 1], 
                                partion_pred()); // уплотнение
                n_rays = mid - ptr;              // число лучей для дальнейшей обработки
                if (n_rays == 0)
                {
                    break;
                }
                n_levels[R] = n_rays << 1;       // сохраняем это число
                sum_rays += n_levels[R];

                if (n_levels[R] > n_levels_size[R])   // число лучей больше чем аллоцированная память
                {
                    /* аллоцируем ещё память */
                    if (tree_dev[R] != NULL)
                    {
                        CSC(cudaFree(tree_dev[R]));
                    }
                    if (tree[R] != NULL)
                    {
                        free(tree[R]);
                    }

                    tree[R] = (node *)malloc(n_levels[R] * sizeof(node));
                    if (!tree[R])
                    {
                        printf("Error: Not enough CPU memory to make operation\n");
                        goto FREE;
                    }
                    CSC(cudaMalloc(tree_dev + R, n_levels[R] * sizeof(node)));

                    n_levels_size[R] = n_levels[R];
                }
                auto render_grid = optimal_grid(n_rays, blocks, threads);
                render_kernel<<<render_grid.first, render_grid.second>>>
                               (n_rays,
                                polygons_dev, normals_dev, colors_dev, krs_dev, krefs_dev, n_polygons,
                                light_points_dev, light_colors_dev, n_lights,
                                tex_dev, wtex, htex, ax_dev, ay_dev, tex_start,
                                tree_dev[R - 1], tree_dev[R]);
                DCSC(cudaDeviceSynchronize());
                DCSC(cudaGetLastError());
            }

            /* Копирование и сортировка */
            for (k = 0; k < R - 1; k++)
            {
                ptr = thrust::device_pointer_cast(tree_dev[k]);
                sort(thrust::device, ptr, ptr + n_levels[k], sort_greater());
                CSC(cudaMemcpy(tree[k], tree_dev[k], n_levels[k] * sizeof(node), cudaMemcpyDeviceToHost));
            }
            /* На последнем уровне уже отсортировано */
            CSC(cudaMemcpy(tree[k], tree_dev[k], n_levels[k] * sizeof(node), cudaMemcpyDeviceToHost));
            
            int size = hs,
                block_size = (size - 1) / omp_threads + 1;

#pragma omp parallel num_threads(omp_threads) private(y, x)
            {
                int id = omp_get_thread_num(),
                    nthrs = omp_get_num_threads();

                int begin = id * block_size,
                    end = MIN((id + 1) * block_size, size);

                for (y = begin; y < end; y++)
                {
                    for (x = 0; x < ws; x++)
                    {
                        pixels_SSAA[(hs - 1 - y) * ws + x] = color2bytes(calculate_color(y * ws + x, 0, R - 1, tree));
                    }
                }
            }
            std::fill(n_levels + 1, n_levels + 1 + MAX_R, 0);

            if (k_SSAA > 1)
            {
                cudaMemcpy(pixels_SSAA_dev, pixels_SSAA, ws * hs * sizeof(uchar4), cudaMemcpyHostToDevice);
                SSAA_kernel<<<SSAA_grid.first, SSAA_grid.second>>>
                            (pixels_dev, w, h, k_SSAA, k_SSAA, pixels_SSAA_dev);
                DCSC(cudaDeviceSynchronize());
                DCSC(cudaGetLastError());
                cudaMemcpy(pixels, pixels_dev, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost);
            }
            else // k_SSAA == 1
            {
                int size = h,
                    block_size = (size - 1) / omp_threads + 1;

                #pragma omp parallel num_threads(omp_threads) private(y, x)
                {
                    int id = omp_get_thread_num(),
                        nthrs = omp_get_num_threads();

                    int begin = id * block_size,
                        end = MIN((id + 1) * block_size, size);

                    for (y = begin; y < end; y++)
                    {
                        for (x = 0; x < w; x++)
                        {
                            pixels[y * w + x] = pixels_SSAA[y * w + x];
                        }
                    }
                }
            }
            
            sprintf(buff, path2images, i);
            fp = fopen(buff, "wb");
            fwrite(&w, sizeof(int), 1, fp);
            fwrite(&h, sizeof(int), 1, fp);
            fwrite(pixels, sizeof(uchar4), w * h, fp);
            fclose(fp);

            time_end = high_resolution_clock::now();
            time_span = std::chrono::duration_cast<milliseconds>(time_end - time_start);

            printf("%12d|%12.2f|%12d\n",
                   i + 1,
                   (float)time_span.count(),
                   sum_rays);
            fflush(stdout);
        }
        for (i = 0; i < MAX_R + 1; i++)
        {
            free(tree[i]);
            CSC(cudaFree(tree_dev[i]));
        }
        free(tree);
    }
    else // CPU
    {
        for (i = 0; i < frames; i++)
        {
            time_start = high_resolution_clock::now();

            t = (float)i * dt;

            rc = r0c + Arc * sinf(omegarc * t + prc);
            zc = z0c + Azc * sinf(omegazc * t + pzc);
            phic = phi0c + omegaphic * t;

            rv = r0v + Arv * sinf(omegarv * t + prv);
            zv = z0v + Azv * sinf(omegazv * t + pzv);
            phiv = phi0v + omegaphiv * t;

            pc = { rc * cosf(phic), rc * sinf(phic), zc };
            pv = { rv * cosf(phiv), rv * sinf(phiv), zv };

            render(pc, pv, ws, hs, fov, MAX_R, polygons, normals, colors, krs, krefs, n_polygons,
                   light_points, light_colors, n_lights,
                   pixels_SSAA,
                   tex, wtex, htex, ax, ay, tex_start,
                   omp_threads);

            SSAA(pixels, w, h, k_SSAA, k_SSAA, pixels_SSAA, omp_threads);
            
            sprintf(buff, path2images, i);
            fp = fopen(buff, "wb");
            fwrite(&w, sizeof(int), 1, fp);
            fwrite(&h, sizeof(int), 1, fp);
            fwrite(pixels, sizeof(uchar4), w * h, fp);
            fclose(fp);

            time_end = high_resolution_clock::now();
            time_span = std::chrono::duration_cast<milliseconds>(time_end - time_start);

            printf("%12d|%12.2f|%12s\n",
                   i + 1,
                   (float)time_span.count(),
                   "----------");
            fflush(stdout);
        }
    }

FREE:
    
    cudaFree(MEM_GPU);

FREE2:

    free(pixels_SSAA);

FREE1:

    free(MEM_CPU);

    return 0;
}
