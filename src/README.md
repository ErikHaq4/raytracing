# Сборка и запуск 

#### Сборка для Win10 x64

1. Скачать данный репозиторий;

2. Установить Microsoft Visual Studio (VS) версии не менее 2019 года;

3. Установить CUDA sdk для VS;

4. Открыть файл src/cuda/RayTracing.sln из репозитория в VS;

5. Собрать проект в конфигурации Release, x64. После этого в директории src\cuda\x64\Release должны появиться файлы в т.ч. RayTracing.exe. Для теста, что этот шаг прошёл успешно, можно запустить проект в VS. Консольный вывод должен быть следующим:

   ```powershell
   GPU run
   omp_threads = 16
   using NVIDIA GeForce RTX 3090
   n_polygons = 382
   frame_number| time2render| num of rays
              1|       62.00|     1409710
              2|       44.00|     1199510
              3|       37.00|     1062492
              4|       35.00|      986040
              5|       33.00|      937318
              6|       33.00|      901988
   ...
   ```

6. Если требуется api на Python, то нужно установить пакеты для этого языка: numpy, cv2, struct, ctypes, pathlib, multiprocessing, subprocess;

7. Если требуется создание анимации, то нужно дополнительно установить консольную утилиту ffmpeg. Это можно сделать, например, с помощью пакетного менеджера winget для терминала windows:

   ```powershell
   $ winget install ffmpeg
   ```

8. Для теста работы api на Python запустить скрипт src/scripts/raytracing.py без параметров:

   ```powershell
   $ python .\src\scripts\raytracing.py
   ```

   Вывод должен быть следующим:

   ```powershell
   Usage: python raytracing.py key [arg1] [arg2] ...
   
   key = -conv:
       arg1 = -int2png
           arg2 = -all                Конвертация всех кадров из внутреннего формата *.dat в формат *.png.
                                      Файлы *.dat должны находиться в папке cuda/frames_dat, выходные
                                      файлы *.png будут находиться в папке cuda/frames_png
   ...
   ```

#### Запуск и примеры работы

Тут идёт описание запуска, ключей, примеров работы
