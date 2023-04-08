TARGET = a.exe
COMPILER = nvcc
FLAGS = -std=c++11 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -Xcompiler -fopenmp
all :
	$(COMPILER) $(FLAGS) kernel.cu -o $(TARGET)

clean :
	rm -f $(TARGET)