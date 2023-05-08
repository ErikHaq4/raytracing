# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:57:09 2020

@author: stife
"""

#%% Либы

import sys
import os
import numpy as np
import cv2
from sys import argv
import struct
import ctypes
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool
import subprocess as sp

#%% Функции

def png2int(from_path, to_path):
    im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGBA)

    w = im.shape[1]
    h = im.shape[0]

    buff = ctypes.create_string_buffer(4 * w * h)
    struct.pack_into(">" + str(4 * w * h) + "B", buff, 0, *im.reshape(-1))

    with open(to_path, "wb") as file:
        file.write(struct.pack("ii", w, h))
        file.write(buff.raw)

def int2png(from_path, to_path):

    w = h = 0
    buff = None
    with open(from_path, "rb") as file:
        w, h = struct.unpack("ii", file.read(8))
        buff = ctypes.create_string_buffer(4 * w * h)
        file.readinto(buff)

    im = struct.unpack_from(">" + str(4 * w * h) + "B", buff)
    im = np.array(im, dtype=np.uint8).reshape((h, w, 4))

    cv2.imwrite(to_path, cv2.cvtColor(im, cv2.COLOR_BGR2RGBA))

def int2png_all(datdir, pngdir, threads=-1):
    """
    Конвертация в png
    """

    # Удаление старых png
    for file in filter(lambda x : x.endswith(".png"), os.listdir(pngdir)):
        os.remove(os.path.join(pngdir, file))

    files = filter(lambda x : x.endswith(".dat"), os.listdir(datdir))

    jobs = []

    for i, file in enumerate(files):

        name = file[0:len(file)-4]

        jobs.append((os.path.join(datdir, file), name+".png"))

    if len(jobs) == 0:

        print("ERROR: no *.dat files in %s" % datdir)
        return

    # смена директории,
    # поскольку cv2.imwrite не работает с путями, содержащими не ASCII символы

    script_dir = os.getcwd()
    os.chdir(pngdir)

    if threads == -1:
        threads = mp.cpu_count()

    threads = min(len(jobs), threads)

    print("Converting *.dat to *.png ... ", end="")
    sys.stdout.flush()

    with Pool(processes=threads) as pool:
        pool.starmap(int2png, jobs, chunksize=((len(jobs) - 1) // threads + 1))

    print("OK")
    print("Created %d png frames in %s" % (len(jobs), pngdir))

    # Возврат директории

    os.chdir(script_dir)

#%% MAIN

DESC = """
Usage: python raytracing.py key [arg1] [arg2] ...

key = -conv:
    arg1 = -int2png
        arg2 = -all                Конвертация всех кадров из внутреннего формата *.dat в формат *.png.
                                   Файлы *.dat должны находиться в папке cuda/frames_dat (или 
                                   в пользовательской папке, указанной после опционального ключа -ifolder),
                                   выходные файлы *.png будут находиться в папке cuda/frames_png (или 
                                   в пользовательской папке, указанной после опционального ключа -ofolder)
                                   ==============================================================
        arg2 = path2file1
            arg3 = path2file2      Конвертация одного файла path2file1 из внутреннего формата *.dat
                                   в формат *.(png|jpg|tiff) и сохранение его в path2file2
                                   ==============================================================
    arg1 = -png2int
        arg2 = path2file1
            arg3 = path2file2      Конвертация одного файла path2file1 из формата *.(png|jpg|tiff) во внутренний
                                   формат *.dat и сохранение его в path2file2
                                   ==============================================================
key = -video
    [arg1] = path2video
        [arg2] = fps               Создание видео с помощью кадров из папки cuda/frames_png (или пользовательской папке,
                                   указанной после опционального ключа -ifolder) с выходным
                                   именем и местоположением path2video и числом кадров в секунду fps.
                                   Если опциональный параметр path2video не указан, то path2video = res.mp4
                                   и опциональный параметр fps = 10. Для работы данного режима
                                   необходима утилита ffmpeg, добавленная в переменную среды Path.
                                   Поддерживаются выходные форматы *.(mp4|gif|avi)
                                   ==============================================================

key = -render                      Рендер сцены. Далее идёт описание параметров, все их них опциональны:
                                   если не указаны, то используются значения по умолчанию.

    [argv(i)] = -gpu               Использовать gpu и cpu. По умолчанию включён

    [argv(i)] = -cpu               Использовать только сpu. По умолчанию выключен

    [arg(i)] = -nframes            Число выходных кадров. nframes > 1
        arg(i+1) = nframes

    [arg(i)] = -frames_dir         Папка для хранения выходных кадров. Строка.
        arg(i+1) = frames_dir      Абсолютный или относительный путь, начиная с директории cuda.
                                   Путь не должен содержать кириллицы, пустых символов

    [arg(i)] = -w                  Ширина выходного кадра. w > 0
        arg(i+1) = w
    [arg(i)] = -h                  Высота выходного кадра. h > 0
        arg(i+1) = h
    [arg(i)] = -fov                Угол обзора в градуса. Целое число. 1 <= fov <= 180
        arg(i+1) = fov
    [arg(i)] = -camera_c           Группы параметров camera_c и camera_v определяют
        arg(i+1) = r0c             закон движения камеры.
        arg(i+2) = z0c             camera_c - это параметры перемещения камеры, а
        arg(i+3) = phi0c           camera_v - это параметры направления камеры.
        arg(i+4) = Arc
        arg(i+5) = Azc             Точка C положения камеры в цилиндрических координатах (r, phi, z)
        arg(i+6) = omegarc         в момент времени t из [0, 2pi] есть
        arg(i+7) = omegazc
        arg(i+8) = omegaphic       rc(t) = r0c + Arc * sin(omegarc * t + prc)
        arg(i+9) = prc             zc(t) = z0c + Azc * sin(omegazc * t + pzc)
        arg(i+10) = pzc            phic(t) = phi0c + omegaphic * t
    [arg(i)] = -camera_v
        arg(i+1) = r0v             Точка V направления камеры в цилиндрических координатах (r, phi, z)
        arg(i+2) = z0v             в момент времени t из [0, 2pi] есть
        arg(i+3) = phi0v
        arg(i+4) = Arv             rv(t) = r0v + Arv * sin(omegarv * t + prv)
        arg(i+5) = Azv             zv(t) = z0v + Azv * sin(omegazv * t + pzv)
        arg(i+6) = omegarv         phiv(t) = phi0v + omegaphiv * t
        arg(i+7) = omegazv
        arg(i+8) = omegaphiv       Момент времени t дискретизирован от 0 до 2pi с шагом 2pi / (nframes - 1)
        arg(i+9) = prv             Группы параметров camera_c и camera_v - это дробные числа float
        arg(i+10) = pzv
    [arg(i)] = -cube               Параметры КУБА.
        arg(i+1) = x               Координата центра по x.
        arg(i+2) = y               Координата центра по y.
        arg(i+3) = z               Координата центра по z.
        arg(i+4) = R               Радиус описанной окружности. R > 0
        arg(i+5) = r               Цветовая координата r. 0 <= r <= 1
        arg(i+6) = g               Цветовая координата g. 0 <= g <= 1
        arg(i+7) = b               Цветовая координата b. 0 <= b <= 1
        arg(i+8) = kr              Коэффициент отражения. 0 <= kr <= 1
        arg(i+9) = kref            Коэффицент прозрачности. 0 <= kref <= 1, kr + kref <= 1
        arg(i+10) = nobj           Число правильных геометрических тел на ребре, целое число. nobj >= 0
    [arg(i)] = -dodecahedron       Параметры ДОДЕКАЭДРА.
        arg(i+1) = x               Координата центра по x.
        arg(i+2) = y               Координата центра по y.
        arg(i+3) = z               Координата центра по z.
        arg(i+4) = R               Радиус описанной окружности. R > 0
        arg(i+5) = r               Цветовая координата r. 0 <= r <= 1
        arg(i+6) = g               Цветовая координата g. 0 <= g <= 1
        arg(i+7) = b               Цветовая координата b. 0 <= b <= 1
        arg(i+8) = kr              Коэффициент отражения. 0 <= kr <= 1
        arg(i+9) = kref            Коэффицент прозрачности. 0 <= kref <= 1, kr + kref <= 1
        arg(i+10) = nobj           Число правильных геометрических тел на ребре, целое число. nobj >= 0
    [arg(i)] = -icosahedron        Параметры ИКОСАЭДРА.
        arg(i+1) = x               Координата центра по x.
        arg(i+2) = y               Координата центра по y.
        arg(i+3) = z               Координата центра по z.
        arg(i+4) = R               Радиус описанной окружности. R > 0
        arg(i+5) = r               Цветовая координата r. 0 <= r <= 1
        arg(i+6) = g               Цветовая координата g. 0 <= g <= 1
        arg(i+7) = b               Цветовая координата b. 0 <= b <= 1
        arg(i+8) = kr              Коэффициент отражения. 0 <= kr <= 1
        arg(i+9) = kref            Коэффицент прозрачности. 0 <= kref <= 1, kr + kref <= 1
        arg(i+10) = nobj           Число правильных геометрических тел на ребре, целое число. nobj >= 0
    [arg(i)] = -floor              Параметры текстурированного ПОЛА.
        arg(i+1)  = a.x            
        arg(i+2)  = a.y            Геометрическое положение пола задаётся четырьмя точками краёв в декартовой системе
        arg(i+3)  = a.z            координат: A, B, C, D. Путь к пользовательской текстуре во внутреннем формате *.dat задаётся
        arg(i+4)  = b.x            параметром-строкой path2tex. Для текстур в формате png, jpeg, diff и прочих используй
        arg(i+5)  = b.y            данный скрипт с ключом -conv (см. описание).
        arg(i+6)  = b.z
        arg(i+7)  = c.x            Текстура накладывается на пол с помощью билинейной интерполяции
        arg(i+8)  = c.y
        arg(i+9)  = c.z            Вместо текстуры можно использовать простую заливку цветом (r, g, b), но данный режим пока 
        arg(i+10) = d.x            не реализован
        arg(i+11) = d.y
        arg(i+12) = d.z
        arg(i+13) = path2tex       Путь к текстуре во внутреннем формате *.dat. Абсолютный или относительный путь, начиная с директории cuda.
                                   Путь не должен содержать кириллицы, пустых символов
        arg(i+14) = r              0 <= r <= 1
        arg(i+15) = g              0 <= g <= 1
        arg(i+16) = b              0 <= b <= 1. Параметры r, g, b пока не используются
        arg(i+17) = kr             Коэффициент отражения. 0 <= kr <= 1
    [arg(i)] = -nlights            Число источников света сцены.
        arg(i+1) = nlights         Целое число, nlights >= 0
        [arg(i+1+1)]  = x          
        [arg(i+1+2)]  = y
        [arg(i+1+3)]  = z          x, y, z - это координаты источника света с номером j. 0 <= j < nlights
        [arg(i+1+4)]  = r          0 <= r <= 1
        [arg(i+1+5)]  = g          0 <= g <= 1
        [arg(i+1+6)]  = b          0 <= b <= 1. (r, g, b) - это цвет источника света с номером j. 0 <= j < nlights
        [arg(i+1+7)] = x
        [arg(i+1+8)] = y
        ...
        [arg(i+1+(nlights-1)*6+6)] = b
    [arg(i)] = -rmax               Максимальная глубина рекурсии. rmax >= 0. 
        arg(i+1) = rmax            Один луч максимально отслеживается rmax + 1 раз при
                                   переотражениях и прохождении насквозь через различные тела. Если луч вылетает за сцену,
                                   то далее он не отслеживается. Большие значения улучшают картинку, но и увеличивают время рендера.
                                   rmax = 0 означает не отслеживать луч после контакта с телом. В этом случае эффектов
                                   отражения и прозрачности не будет
    [arg(i)] = -ssaa               Коэффициент увеличения масштаба рендера со сглаживанием по алгоритму SSAA. Целое число. ssaa >= 1
        arg(i+1) = ssaa            Фактические размеры кадра есть (ssaa * w) x (ssaa * h), который затем ужимается до w x h.
                                   При ssaa > 1 улучшает выходные изображения, но сильно увеличивает время рендера
"""


FOLDER = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))

def main():

    def get_argv(i, default=None):

        if i >= len(argv):

            if default == None:
                print(DESC)
                exit(0)
            else:
                return default

        return argv[i]

    key1 = get_argv(1)

    if key1 == "-conv":

        key2 = get_argv(2)

        if key2 == "-int2png":

            key3 = get_argv(3)

            if key3 == "-all":

                ifolder = os.path.join(FOLDER, "cuda", "frames_dat")
                ofolder = os.path.join(FOLDER, "cuda", "frames_png")

                i = 4

                while i < len(argv):

                    if argv[i] == "-ifolder":
                        ifolder = get_argv(i + 1)
                        i += 2
                    elif argv[i] == "-ofolder":
                        ofolder = get_argv(i + 1)
                        i += 2
                    else:
                        print(DESC)
                        exit(0)

                threads = mp.cpu_count()

                int2png_all(ifolder, ofolder, threads)
            else:

                from_path = key3
                to_path = get_argv(4)

                head, tail = os.path.split(to_path)

                if (len(head) > 0):

                    from_apath = os.path.abspath(from_path)

                    script_dir = os.getcwd()
                    os.chdir(head)

                    int2png(from_apath, tail)

                    os.chdir(script_dir) # Возврат
                else:

                    print("Hi")
                    int2png(from_path, to_path)
                    print("Converted %s -> %s" % (from_path, to_path))

        elif key2 == "-png2int":

            from_path = argv[2]
            to_path = argv[3]

            from_path = get_argv(3)
            to_path = get_argv(4)

            head, tail = os.path.split(from_path)

            if (len(head) > 0):

                to_apath = os.path.abspath(to_path)

                script_dir = os.getcwd()
                os.chdir(head)

                png2int(tail, to_apath)

                os.chdir(script_dir) # Возврат
            else:

                png2int(from_path, to_path)
                print("Converted %s -> %s" % (from_path, to_path))

        else:

            print(DESC)

    elif key1 == "-video":

        name = "res.mp4"
        fps = 10
        ifolder = os.path.join(FOLDER, "cuda", "frames_png")

        readname = False
        i = 2

        while i < len(argv):

            if argv[i] == "-ifolder":
                ifolder = get_argv(i + 1)
                i += 2
            elif not readname:
                name = argv[i]
                i += 1
                readname = True
            else:
                fps = int(argv[i])
                i += 1

        if os.path.exists(name):
            os.remove(name)

        process = sp.Popen(["ffmpeg", "-framerate", str(fps), "-i", os.path.join(ifolder, "%d.png"), name],
                           stdin=sp.PIPE,
                           stdout=sp.PIPE)
        answer = process.communicate()[0].decode("ascii")
        print(answer)

    elif key1 == "-render":

        path2exe = os.path.join(FOLDER, "cuda", "x64", "Release", "RayTracing.exe")

        path2default = os.path.join(FOLDER, "cuda", "in.txt")

        for file in filter(lambda x : x.endswith(".dat"), os.listdir(os.path.join(FOLDER, "cuda", "frames_dat"))):
            os.remove(os.path.join(FOLDER, "cuda", "frames_dat", file))

        # Чтение параметров по умолчанию

        gpu = True

        with open(path2default, "r", encoding="ascii") as fp:

            def sreadline():

                line = ""

                while len(line) == 0:
                    line = fp.readline().strip()

                return line

            nframes = int(sreadline())
            frames_dir = sreadline()

            line = sreadline().split()
            w, h, fov = int(line[0]), int(line[1]), int(line[2])

            line = sreadline().split()
            camera_params_c = [float(p) for p in line]

            line = sreadline().split()
            camera_params_v = [float(p) for p in line]

            line = sreadline().split()
            cube_params = [float(p) for p in line[0:len(line)-1]] + [int(line[-1])]

            line = sreadline().split()
            dodecahedron_params = [float(p) for p in line[0:len(line)-1]] + [int(line[-1])]

            line = sreadline().split()
            icosahedron_params = [float(p) for p in line[0:len(line)-1]] + [int(line[-1])]

            line = sreadline().split()
            floor_params = [float(p) for p in line[0:12]] + [line[12]] + [float(p) for p in line[13:17]]

            nlights = int(sreadline())
            lights_params = np.empty((nlights, 6), dtype=np.float64)

            for i in range(nlights):
                line = sreadline().split()
                lights_params[i,:] = list(map(float, line))

            line = sreadline().split()
            RMAX, SSAA = int(line[0]), int(line[1])

            # Чтение пользовательских параметров

            i = 2

            while i < len(argv):

                key = argv[i]

                if key == "-gpu":

                    gpu = True

                    i += 1
                elif key == "-cpu":

                    gpu = False

                    i += 1
                elif key == "-nframes":

                    nframes = int(get_argv(i + 1))

                    i += 2
                elif key == "-frames_dir":

                    frames_dir = get_argv(i + 1)

                    i += 2
                elif key == "-w":

                    w = int(get_argv(i + 1))

                    i += 2
                elif key == "-h":

                    h = int(get_argv(i + 1))

                    i += 2
                elif key == "-fov":

                    fov = int(get_argv(i + 1))

                    i += 2
                elif key == "-camera_c":

                    for j in range(10):
                        camera_params_c[j] = float(argv[i + 1 + j])

                    i += 11

                elif key == "-camera_v":

                    for j in range(10):
                        camera_params_v[j] = float(argv[i + 1 + j])

                    i += 11
                elif key == "-cube":

                    for j in range(9):
                        cube_params[j] = float(argv[i + 1 + j])

                    cube_params[9] = int(argv[i + 1 + 9])

                    i += 11
                elif key == "-dodecahedron":

                    for j in range(9):
                        dodecahedron_params[j] = float(argv[i + 1 + j])

                    dodecahedron_params[9] = int(argv[i + 1 + 9])

                    i += 11
                elif key == "-icosahedron":

                    for j in range(9):
                        icosahedron_params[j] = float(argv[i + 1 + j])

                    icosahedron_params[9] = int(argv[i + 1 + 9])

                    i += 11
                elif key == "-floor":

                    for j in range(12):
                        floor_params[j] = float(argv[i + 1 + j])

                    floor_params[12] = argv[i + 1 + 12]

                    for j in range(4):
                        floor_params[12 + 1 + j] = float(argv[i + 1 + 12 + 1 + j])

                    i += 1 + 12 + 1 + 4
                elif key == "-nlights":

                    nlights = int(get_argv(i + 1))

                    lights_params = np.empty((nlights, 6), dtype=np.float64)

                    for j in range(nlights):
                        for k in range(6):
                            lights_params[j, k] = float(argv[i + 2 + j * 6 + k])
                    
                    i += 1 + 1 + nlights * 6
                elif key == "-rmax":

                    RMAX = int(get_argv(i + 1))

                    i += 2

                elif key == "-ssaa":

                    SSAA = int(get_argv(i + 1))

                    i += 2

                else:

                    print(DESC)

        print("Использовать GPU: %s" % ("true" if gpu else "false"))
        print("Число кадров: %d" % nframes)
        print("Папка с выходными кадрами во внутреннем формате: %s" % frames_dir)
        print("Разрешение: %dx%d" % (w, h))
        print("Угол обзора: %d" % fov)
        print("Параметры движения камеры: %s" % str(camera_params_c))
        print("Параметры направления камеры: %s" % str(camera_params_v))
        print("Параметры куба: %s" % str(cube_params))
        print("Параметры додекаэдра: %s" % str(dodecahedron_params))
        print("Параметры икосаэдра: %s" % str(icosahedron_params))
        print("Параметры пола: %s" % str(floor_params))
        print("Число источников света: %d" % nlights)
        for i in range(nlights):
            print("Источник %d: %s" % (i + 1, str(lights_params[i])))
        print("Максимальная глубина рекурсии: %d" % RMAX)
        print("Коэффицент SSAA: %d" % SSAA)
        print("Разрешение рендера: %dx%d" % (w * SSAA, h * SSAA))

        string_params = \
("\
%d \
%s \
%d %d % d \
%f %f %f %f %f %f %f %f %f %f \
%f %f %f %f %f %f %f %f %f %f \
%f %f %f %f %f %f %f %f %f %d \
%f %f %f %f %f %f %f %f %f %d \
%f %f %f %f %f %f %f %f %f %d \
%f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %f \
%d \
"       + \
"\
%f %f %f %f %f %f \
"       * nlights + \
"\
%d %d\n\
")       % \
        (nframes,
         frames_dir,
         w, h, fov,
         *camera_params_c,
         *camera_params_v,
         *cube_params,
         *dodecahedron_params,
         *icosahedron_params,
         *floor_params,
         nlights,
         *lights_params.ravel(),
         RMAX, SSAA)

        script_dir = os.getcwd()
        os.chdir(os.path.join(FOLDER, "cuda"))

        p = sp.Popen([path2exe, "--gpu" if gpu else "--cpu"],
                     stdin=sp.PIPE,
                     stdout=sp.PIPE)

        p.stdin.write(string_params.encode("ascii"))
        p.stdin.close()

        while True:
            line = p.stdout.readline()
            if not line:
              break
            print(line.decode("ascii").rstrip())

        os.chdir(script_dir)

    else:
        print(DESC)


if __name__ == "__main__":
    main()