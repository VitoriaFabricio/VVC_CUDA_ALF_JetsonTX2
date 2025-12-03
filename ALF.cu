#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

// Headers para monitoramento de tempo e energia
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

// Definições do Filtro ALF
#define COEFF_SCALE_BITS 6 
#define FILTER_TILE_W 32
#define FILTER_TILE_H 32
#define BORDER_SIZE 3

// Configurações da Classificação
#define CLASSIFY_TILE_SIZE 32
#define CLASSIFY_BLOCK_SIZE 4
#define LAPLACIAN_HALO 1 
#define WINDOW_SIZE 8 

#define NUM_CLASSES 25
#define NUM_COEFFS_PER_CLASS 13
#define BIT_DEPTH 10
#define MAX_PIXEL_VAL ((1 << BIT_DEPTH) - 1) // 1023 para 10-bit

// Constantes em memória constante da GPU
__constant__ int g_th[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 };
__constant__ int g_alfLuma_dy_filter[12] = {-3, -3, -3, -2, -2, -2, -2, -1, -1, -1, 0, 0};
__constant__ int g_alfLuma_dx_filter[12] = {0, 1, 2, -1, 0, 1, 2, -1, 0, 1, -1, -2};

// Coeficientes fixos (VTM)
const short m_fixedFilterSetCoeff[64][13] = {
    { 0, 0, 2, -3, 1, -4, 1, 7, -1, 1, -1, 5, 0 },    { 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 2, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0 },
    { 2, 2, -7, -3, 0, -5, 13, 22, 12, -3, -3, 17, 0 }, { -1, 0, 6, -8, 1, -5, 1, 23, 0, 2, -5, 10, 0 },
    { 0, 0, -1, -1, 0, -1, 2, 1, 0, 0, -1, 4, 0 },    { 0, 0, 3, -11, 1, 0, -1, 35, 5, 2, -9, 9, 0 },
    { 0, 0, 8, -8, -2, -7, 4, 4, 2, 1, -1, 25, 0 },   { 0, 0, 1, -1, 0, -3, 1, 3, -1, 1, -1, 3, 0 },
    { 0, 0, 3, -3, 0, -6, 5, -1, 2, 1, -4, 21, 0 },   { -7, 1, 5, 4, -3, 5, 11, 13, 12, -8, 11, 12, 0 },
    { -5, -3, 6, -2, -3, 8, 14, 15, 2, -7, 11, 16, 0 }, { 2, -1, -6, -5, -2, -2, 20, 14, -4, 0, -3, 25, 0 },
    { 3, 1, -8, -4, 0, -8, 22, 5, -3, 2, -10, 29, 0 },  { 2, 1, -7, -1, 2, -11, 23, -5, 0, 2, -10, 29, 0 },
    { -6, -3, 8, 9, -4, 8, 9, 7, 14, -2, 8, 9, 0 },     { 2, 1, -4, -7, 0, -8, 17, 22, 1, -1, -4, 23, 0 },
    { 3, 0, -5, -7, 0, -7, 15, 18, -5, 0, -5, 27, 0 },  { 2, 0, 0, -7, 1, -10, 13, 13, -4, 2, -7, 24, 0 },
    { 3, 3, -13, 4, -2, -5, 9, 21, 25, -2, -3, 12, 0 }, { -5, -2, 7, -3, -7, 9, 8, 9, 16, -2, 15, 12, 0 },
    { 0, -1, 0, -7, -5, 4, 11, 11, 8, -6, 12, 21, 0 },  { 3, -2, -3, -8, -4, -1, 16, 15, -2, -3, 3, 26, 0 },
    { 2, 1, -5, -4, -1, -8, 16, 4, -2, 1, -7, 33, 0 },  { 2, 1, -4, -2, 1, -10, 17, -2, 0, 2, -11, 33, 0 },
    { 1, -2, 7, -15, -16, 10, 8, 8, 20, 11, 14, 11, 0 },{ 2, 2, 3, -13, -13, 4, 8, 12, 2, -3, 16, 24, 0 },
    { 1, 4, 0, -7, -8, -4, 9, 9, -2, -2, 8, 29, 0 },    { 1, 1, 2, -4, -1, -6, 6, 3, -1, -1, -3, 30, 0 },
    { -7, 3, 2, 10, -2, 3, 7, 11, 19, -7, 8, 10, 0 },   { 0, -2, -5, -3, -2, 4, 20, 15, -1, -3, -1, 22, 0 },
    { 3, -1, -8, -4, -1, -4, 22, 8, -4, 2, -8, 28, 0 }, { 0, 3, -14, 3, 0, 1, 19, 17, 8, -3, -7, 20, 0 },
    { 0, 2, -1, -8, 3, -6, 5, 21, 1, 1, -9, 13, 0 },    { -4, -2, 8, 20, -2, 2, 3, 5, 21, 4, 6, 1, 0 },
    { 2, -2, -3, -9, -4, 2, 14, 16, 3, -6, 8, 24, 0 },  { 2, 1, 5, -16, -7, 2, 3, 11, 15, -3, 11, 22, 0 },
    { 1, 2, 3, -11, -2, -5, 4, 8, 9, -3, -2, 26, 0 },  { 0, -1, 10, -9, -1, -8, 2, 3, 4, 0, 0, 29, 0 },
    { 1, 2, 0, -5, 1, -9, 9, 3, 0, 1, -7, 20, 0 },      { -2, 8, -6, -4, 3, -9, -8, 45, 14, 2, -13, 7, 0 },
    { 1, -1, 16, -19, -8, -4, -3, 2, 19, 0, 4, 30, 0 }, { 1, 1, -3, 0, 2, -11, 15, -5, 1, 2, -9, 24, 0 },
    { 0, 1, -2, 0, 1, -4, 4, 0, 0, 1, -4, 7, 0 },       { 0, 1, 2, -5, 1, -6, 4, 10, -2, 1, -4, 10, 0 },
    { 3, 0, -3, -6, -2, -6, 14, 8, -1, -1, -3, 31, 0 }, { 0, 1, 0, -2, 1, -6, 5, 1, 0, 1, -5, 13, 0 },
    { 3, 1, 9, -19, -21, 9, 7, 6, 13, 5, 15, 21, 0 },  { 2, 4, 3, -12, -13, 1, 7, 8, 3, 0, 12, 26, 0 },
    { 3, 1, -8, -2, 0, -6, 18, 2, -2, 3, -10, 23, 0 },  { 1, 1, -4, -1, 1, -5, 8, 1, -1, 2, -5, 10, 0 },
    { 0, 1, -1, 0, 0, -2, 2, 0, 0, 1, -2, 3, 0 },       { 1, 1, -2, -7, 1, -7, 14, 18, 0, 0, -7, 21, 0 },
    { 0, 1, 0, -2, 0, -7, 8, 1, -2, 0, -3, 24, 0 },     { 0, 1, 1, -2, 2, -10, 10, 0, -2, 1, -7, 23, 0 },
    { 0, 2, 2, -11, 2, -4, -3, 39, 7, 1, -10, 9, 0 },  { 1, 0, 13, -16, -5, -6, -1, 8, 6, 0, 6, 29, 0 },
    { 1, 3, 1, -6, -4, -7, 9, 6, -3, -2, 3, 33, 0 },    { 4, 0, -17, -1, -1, 5, 26, 8, -2, 3, -15, 30, 0 },
    { 0, 1, -2, 0, 2, -8, 12, -6, 1, 1, -6, 16, 0 },    { 0, 0, 0, -1, 1, -4, 4, 0, 0, 0, -3, 11, 0 },
    { 0, 1, 2, -8, 2, -6, 5, 15, 0, 2, -7, 9, 0 },      { 1, -1, 12, -15, -7, -2, 3, 6, 6, -1, 7, 30, 0 },
};

const int m_classToFilterMapping[16][25] = {
    {  8,   2,   2,   2,   3,   4,  53,   9,   9,  52,   4,   4,   5,   9,   2,   8,  10,   9,   1,   3,  39,  39,  10,   9,  52 },
    { 11,  12,  13,  14,  15,  30,  11,  17,  18,  19,  16,  20,  20,   4,  53,  21,  22,  23,  14,  25,  26,  26,  27,  28,  10 },
    { 16,  12,  31,  32,  14,  16,  30,  33,  53,  34,  35,  16,  20,   4,   7,  16,  21,  36,  18,  19,  21,  26,  37,  38,  39 },
    { 35,  11,  13,  14,  43,  35,  16,   4,  34,  62,  35,  35,  30,  56,   7,  35,  21,  38,  24,  40,  16,  21,  48,  57,  39 },
    { 11,  31,  32,  43,  44,  16,   4,  17,  34,  45,  30,  20,  20,   7,   5,  21,  22,  46,  40,  47,  26,  48,  63,  58,  10 },
    { 12,  13,  50,  51,  52,  11,  17,  53,  45,   9,  30,   4,  53,  19,   0,  22,  23,  25,  43,  44,  37,  27,  28,  10,  55 },
    { 30,  33,  62,  51,  44,  20,  41,  56,  34,  45,  20,  41,  41,  56,   5,  30,  56,  38,  40,  47,  11,  37,  42,  57,   8 },
    { 35,  11,  23,  32,  14,  35,  20,   4,  17,  18,  21,  20,  20,  20,   4,  16,  21,  36,  46,  25,  41,  26,  48,  49,  58 },
    { 12,  31,  59,  59,   3,  33,  33,  59,  59,  52,   4,  33,  17,  59,  55,  22,  36,  59,  59,  60,  22,  36,  59,  25,  55 },
    { 31,  25,  15,  60,  60,  22,  17,  19,  55,  55,  20,  20,  53,  19,  55,  22,  46,  25,  43,  60,  37,  28,  10,  55,  52 },
    { 12,  31,  32,  50,  51,  11,  33,  53,  19,  45,  16,   4,   4,  53,   5,  22,  36,  18,  25,  43,  26,  27,  27,  28,  10 },
    {  5,   2,  44,  52,   3,   4,  53,  45,   9,   3,   4,  56,   5,   0,   2,   5,  10,  47,  52,   3,  63,  39,  10,   9,  52 },
    { 12,  34,  44,  44,   3,  56,  56,  62,  45,   9,  56,  56,   7,   5,   0,  22,  38,  40,  47,  52,  48,  57,  39,  10,   9 },
    { 35,  11,  23,  14,  51,  35,  20,  41,  56,  62,  16,  20,  41,  56,   7,  16,  21,  38,  24,  40,  26,  26,  42,  57,  39 },
    { 33,  34,  51,  51,  52,  41,  41,  34,  62,   0,  41,  41,  56,   7,   5,  56,  38,  38,  40,  44,  37,  42,  57,  39,  10 },
    { 16,  31,  32,  15,  60,  30,   4,  17,  19,  25,  22,  20,   4,  53,  19,  21,  22,  46,  25,  55,  26,  48,  63,  58,  55 },
};

void checkCUDAError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define gpuErrchk(ans) { checkCUDAError((ans), __FILE__, __LINE__); }

__device__ inline int clip(int x, int minVal, int maxVal) {
    return max(minVal, min(x, maxVal));
}

// Transformação Geométrica
__device__ inline void transform_coords_filter(int &dx, int &dy, int transformID) {
    if (transformID == 1) { 
        int temp = dx; dx = dy; dy = temp;
    } else if (transformID == 2) { 
        int temp = dx; dx = dy; dy = -temp;
    } else if (transformID == 3) { 
        dx = -dx; dy = -dy; 
    }
}

// Estrutura para Monitoramento de Energia (Jetson)
typedef struct {
    int keep_running;
    unsigned int sample_count;
    unsigned int max_samples;

    unsigned int* gpu_power_samples;
    unsigned int* cpu_power_samples;
} MonitorThreadData;

// Leitura de energia via sysfs (Jetson TX2 INA3221)
long read_power_mW(const char* path) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        static int warning_shown = 0;
        if (!warning_shown) {
            fprintf(stderr, "Aviso: Nao foi possivel abrir o arquivo de energia %s. Tente executar com 'sudo'.\n", path);
            warning_shown = 1;
        }
        return 0;
    }

    long power_mW;
    if (fscanf(file, "%ld", &power_mW) != 1) {
        power_mW = 0;
    }
    fclose(file);
    return power_mW;
}

// Thread que coleta amostras de energia
void* powerMonitorThread(void* arg) {
    MonitorThreadData* data = (MonitorThreadData*)arg;

    // Caminhos para INA3221 na Jetson TX2 (VDD_GPU e VDD_SYS_CPU)
    const char* gpu_power_path = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input";
    const char* cpu_power_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input";

    while (data->keep_running) {
        if (data->sample_count < data->max_samples) {
            data->gpu_power_samples[data->sample_count] = read_power_mW(gpu_power_path);
            data->cpu_power_samples[data->sample_count] = read_power_mW(cpu_power_path);
            data->sample_count++;
        }
        usleep(1000); // 1ms
    }
    return NULL;
}

// Kernel de Classificação (8x8)
__global__ void gradientAndClassificationKernel(const unsigned short* input_image,
                                                unsigned int* classificationMap,
                                                unsigned int* transformMap,
                                                unsigned int height, unsigned int width) {
    

    const int SMEM_DIM = CLASSIFY_TILE_SIZE + WINDOW_SIZE; // 40x40
    const int INPUT_DIM = SMEM_DIM + 2 * LAPLACIAN_HALO;   // 42x42

    __shared__ unsigned int s_src[INPUT_DIM][INPUT_DIM];
    
    // Matrizes para armazenar os gradientes calculados 
    __shared__ unsigned int s_grad_v[SMEM_DIM][SMEM_DIM];
    __shared__ unsigned int s_grad_h[SMEM_DIM][SMEM_DIM];
    __shared__ unsigned int s_grad_d0[SMEM_DIM][SMEM_DIM];
    __shared__ unsigned int s_grad_d1[SMEM_DIM][SMEM_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int tile_base_x = blockIdx.x * CLASSIFY_TILE_SIZE;
    int tile_base_y = blockIdx.y * CLASSIFY_TILE_SIZE;

    // 1. CARGA PARA SHARED MEMORY
    for (int i = ty; i < INPUT_DIM; i += blockDim.y) {
        for (int j = tx; j < INPUT_DIM; j += blockDim.x) {
            int global_x = tile_base_x + j - LAPLACIAN_HALO;
            int global_y = tile_base_y + i - LAPLACIAN_HALO;

            global_x = max(0, min((int)width - 1, global_x));
            global_y = max(0, min((int)height - 1, global_y));

            s_src[i][j] = (unsigned int)input_image[global_y * width + global_x];
        }
    }
    __syncthreads();

    // 2. CALCULO DOS GRADIENTES (
    for (int i = ty; i < SMEM_DIM; i += blockDim.y) {
        for (int j = tx; j < SMEM_DIM; j += blockDim.x) {
            // Apenas posições pares calculam gradiente 
            if ((i % 2 == 0) && (j % 2 == 0)) {
                int r = i + LAPLACIAN_HALO;
                int c = j + LAPLACIAN_HALO;
                int pix = (int)s_src[r][c];
                
                unsigned int gv = abs((2 * pix) - (int)s_src[r - 1][c] - (int)s_src[r + 1][c]);
                unsigned int gh = abs((2 * pix) - (int)s_src[r][c - 1] - (int)s_src[r][c + 1]);
                unsigned int gd0 = abs((2 * pix) - (int)s_src[r - 1][c - 1] - (int)s_src[r + 1][c + 1]);
                unsigned int gd1 = abs((2 * pix) - (int)s_src[r - 1][c + 1] - (int)s_src[r + 1][c - 1]);

                s_grad_v[i][j]  = gv;
                s_grad_h[i][j]  = gh;
                s_grad_d0[i][j] = gd0;
                s_grad_d1[i][j] = gd1;
            } else {
                s_grad_v[i][j] = 0; s_grad_h[i][j] = 0; s_grad_d0[i][j] = 0; s_grad_d1[i][j] = 0;
            }
        }
    }
    __syncthreads();

    // 3. CLASSIFICAÇÃO (
    int gx = tile_base_x + tx;
    int gy = tile_base_y + ty;

    if (tx < CLASSIFY_TILE_SIZE && ty < CLASSIFY_TILE_SIZE && gx < width && gy < height) {
        if (tx % CLASSIFY_BLOCK_SIZE == 0 && ty % CLASSIFY_BLOCK_SIZE == 0) {
            unsigned int sum_v = 0, sum_h = 0, sum_d0 = 0, sum_d1 = 0;

            // Soma janela 8x8 
            #pragma unroll
            for (int y_off = 0; y_off < 8; y_off += 2) {
                for (int x_off = 0; x_off < 8; x_off += 2) {
                    sum_v  += s_grad_v[ty + y_off][tx + x_off];
                    sum_h  += s_grad_h[ty + y_off][tx + x_off];
                    sum_d0 += s_grad_d0[ty + y_off][tx + x_off];
                    sum_d1 += s_grad_d1[ty + y_off][tx + x_off];
                }
            }

            unsigned int hv1, hv0, d1, d0;
            int mainDirection, secondaryDirection, dirTempHV, dirTempD;

            if (sum_v > sum_h) { hv1 = sum_v; hv0 = sum_h; dirTempHV = 1; }
            else               { hv1 = sum_h; hv0 = sum_v; dirTempHV = 3; }

            if (sum_d0 > sum_d1) { d1 = sum_d0; d0 = sum_d1; dirTempD = 0; }
            else                 { d1 = sum_d1; d0 = sum_d0; dirTempD = 2; }

            if ((unsigned long long)d1 * hv0 > (unsigned long long)hv1 * d0) {
                mainDirection = dirTempD;
                secondaryDirection = dirTempHV;
            } else {
                mainDirection = dirTempHV;
                secondaryDirection = dirTempD;
            }

            unsigned int hvd1, hvd0;
            if (mainDirection == dirTempD) { hvd1 = d1; hvd0 = d0; }
            else                           { hvd1 = hv1; hvd0 = hv0; }

            int directionStrength = 0;
            if (hvd1 > 2 * hvd0)      directionStrength = 1;
            if (hvd1 * 2 > 9 * hvd0)  directionStrength = 2;

            unsigned int activity = sum_v + sum_h;
            const int shift = 12; 
            unsigned int activity_quantized = (activity * 64) >> shift;
            unsigned int A = g_th[activity_quantized < 15 ? activity_quantized : 15];

            unsigned int D = 0;
            if (directionStrength > 0) {
                D = (((mainDirection & 0x1) << 1) + directionStrength);
            }
            
            unsigned int final_class = 5 * D + A;
            const int transposeTable[8] = { 0, 1, 0, 2, 2, 3, 1, 3 };
            unsigned int final_transform = transposeTable[mainDirection * 2 + ( secondaryDirection >> 1 )];

            // Escrita na memória global para todo o bloco 4x4
            for (int i = 0; i < CLASSIFY_BLOCK_SIZE; i++) {
                for (int j = 0; j < CLASSIFY_BLOCK_SIZE; j++) {
                    int out_y = gy + i;
                    int out_x = gx + j;
                    if (out_y < height && out_x < width) {
                        int idx = out_y * width + out_x;
                        classificationMap[idx] = final_class;
                        transformMap[idx] = final_transform;
                    }
                }
            }
        }
    }
}

// Kernel do Filtro ALF 
__global__ void AlfFilterLuma_kernel(unsigned short* filteredImage, const unsigned short* recImage,
                                     const unsigned int* classificationMap, const unsigned int* transformMap,
                                     const short* filterCoefficients,
                                     const int* d_clip_values,
                                     int width, int height, int stride)
{
    const int TILE_W_BORDER = FILTER_TILE_W + 2 * BORDER_SIZE;
    const int TILE_H_BORDER = FILTER_TILE_H + 2 * BORDER_SIZE;
    __shared__ unsigned short s_recTile[TILE_H_BORDER][TILE_W_BORDER];

    const int x = blockIdx.x * FILTER_TILE_W + threadIdx.x;
    const int y = blockIdx.y * FILTER_TILE_H + threadIdx.y;

    for (int i = threadIdx.y; i < TILE_H_BORDER; i += blockDim.y) {
        for (int j = threadIdx.x; j < TILE_W_BORDER; j += blockDim.x) {
            int gx = (blockIdx.x * FILTER_TILE_W) + j - BORDER_SIZE;
            int gy = (blockIdx.y * FILTER_TILE_H) + i - BORDER_SIZE;
            gx = max(0, min(gx, width - 1));
            gy = max(0, min(gy, height - 1));
            s_recTile[i][j] = recImage[gy * stride + gx];
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    const int tx_local = threadIdx.x + BORDER_SIZE;
    const int ty_local = threadIdx.y + BORDER_SIZE;

    const unsigned int classID = classificationMap[y * width + x];
    const unsigned short pixel_central = s_recTile[ty_local][tx_local];
    
    const int transformID = transformMap[y * width + x];
    const short* coeffs = &filterCoefficients[classID * NUM_COEFFS_PER_CLASS];
    const int* clips = &d_clip_values[classID * NUM_COEFFS_PER_CLASS];

    int sum = 0;
    
    for (int i = 0; i < 12; ++i) {
        int dx = g_alfLuma_dx_filter[i];
        int dy = g_alfLuma_dy_filter[i];
        
        transform_coords_filter(dx, dy, transformID);
        
        unsigned short p1 = s_recTile[ty_local + dy][tx_local + dx];
        unsigned short p2 = s_recTile[ty_local - dy][tx_local - dx];

        int clip_val = clips[i];
        int diff1 = clip((int)p1 - (int)pixel_central, -clip_val, clip_val);
        int diff2 = clip((int)p2 - (int)pixel_central, -clip_val, clip_val);
        
        sum += coeffs[i] * (diff1 + diff2);
    }
    
    int offset = 1 << (COEFF_SCALE_BITS - 1);
    int filtered_val = (int)pixel_central + ((sum + offset) >> COEFF_SCALE_BITS);

    filtered_val = max(0, min(filtered_val, MAX_PIXEL_VAL));
    filteredImage[y * stride + x] = (unsigned short)filtered_val;
}

// Funções de salvamento
void saveToCSV(const char* filename, const unsigned short* data, int width, int height) {
    FILE* fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Erro ao abrir %s\n", filename); return; }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            fprintf(fp, "%d", data[y * width + x]);
            if (x < width - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void saveMapToCSV(const char* filename, const unsigned int* data, int width, int height) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            fprintf(fp, "%u", data[y * width + x]);
            if (x < width - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    const int width = 1920;
    const int height = 1080;
    
    // Tamanhos para 10-bit (short)
    const size_t size_pixels_ushort = width * height * sizeof(unsigned short);
    const size_t size_maps = width * height * sizeof(unsigned int);

    printf("Iniciando ALF (Correcao 10-bit) para imagem %dx%d em plataforma Jetson.\n\n", width, height);

    // Preparar Monitoramento de Energia
    MonitorThreadData thread_data;
    thread_data.max_samples = 20000;
    thread_data.gpu_power_samples = (unsigned int*)malloc(thread_data.max_samples * sizeof(unsigned int));
    thread_data.cpu_power_samples = (unsigned int*)malloc(thread_data.max_samples * sizeof(unsigned int));
    
    if (!thread_data.gpu_power_samples || !thread_data.cpu_power_samples) {
        fprintf(stderr, "Falha de alocacao memoria monitor.\n"); return 1;
    }

    cudaEvent_t start_event, stop_event;
    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));

    // Variáveis acumuladoras
    double total_comm_h2d_gpu_time_s = 0, total_comm_d2h_gpu_time_s = 0;
    double total_comm_h2d_gpu_energy_j = 0, total_comm_d2h_gpu_energy_j = 0;
    double total_comm_h2d_cpu_energy_j = 0, total_comm_d2h_cpu_energy_j = 0;
    double total_kernel_gpu_time_s = 0;
    double total_kernel_gpu_energy_j = 0, total_kernel_cpu_energy_j = 0;

    // Alocação Host
    unsigned short* h_recImage = (unsigned short*)malloc(size_pixels_ushort);
    unsigned short* h_filteredImage = (unsigned short*)malloc(size_pixels_ushort);
    unsigned int* h_classificationMap = (unsigned int*)malloc(size_maps);
    unsigned int* h_transformMap = (unsigned int*)malloc(size_maps);

    // Alocação Device
    unsigned short *d_recImage, *d_filteredImage;
    unsigned int *d_classificationMap, *d_transformMap;
    short *d_filterCoefficients;
    int *d_clip_values;

    // Retirado do VTM
    int h_clip_values[NUM_CLASSES * NUM_COEFFS_PER_CLASS] = { 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0, 128, 32, 128, 32, 0, 128, 0, 128, 0, 32, 0, 0, 0 };

    gpuErrchk(cudaMalloc(&d_recImage, size_pixels_ushort));
    gpuErrchk(cudaMalloc(&d_filteredImage, size_pixels_ushort));
    gpuErrchk(cudaMalloc(&d_classificationMap, size_maps));
    gpuErrchk(cudaMalloc(&d_transformMap, size_maps));
    gpuErrchk(cudaMalloc(&d_clip_values, sizeof(h_clip_values)));
    
    // Leitura do CSV
    FILE* file = fopen("original_0.csv", "r");
    if (file == NULL) { fprintf(stderr, "ERRO: Abrir CSV.\n"); return 1; }
    for (int i = 0; i < width * height; ++i) {
        unsigned int temp_val;
        if (fscanf(file, "%u,", &temp_val) != 1) break;
        h_recImage[i] = (unsigned short)temp_val; 
    }
    fclose(file);

    // Setup coeficientes (Host)
    short h_finalCoeffs[NUM_CLASSES * NUM_COEFFS_PER_CLASS];
    int selectedFilterSet = 0;
    for (int classId = 0; classId < NUM_CLASSES; ++classId) {
        if (classId == 0) {
            memset(&h_finalCoeffs[classId * NUM_COEFFS_PER_CLASS], 0, NUM_COEFFS_PER_CLASS * sizeof(short));
            continue;
        }
        int filterId = m_classToFilterMapping[selectedFilterSet][classId];
        memcpy(&h_finalCoeffs[classId * NUM_COEFFS_PER_CLASS], m_fixedFilterSetCoeff[filterId], NUM_COEFFS_PER_CLASS * sizeof(short));
    }
    gpuErrchk(cudaMalloc(&d_filterCoefficients, sizeof(h_finalCoeffs)));
    gpuErrchk(cudaMemcpy(d_filterCoefficients, h_finalCoeffs, sizeof(h_finalCoeffs), cudaMemcpyHostToDevice));

    // Dims
    dim3 threadsClass(CLASSIFY_TILE_SIZE, CLASSIFY_TILE_SIZE);
    dim3 blocksClass((width + CLASSIFY_TILE_SIZE - 1) / CLASSIFY_TILE_SIZE, (height + CLASSIFY_TILE_SIZE - 1) / CLASSIFY_TILE_SIZE);
    
    dim3 threadsFilter(FILTER_TILE_W, FILTER_TILE_H);
    dim3 blocksFilter((width + FILTER_TILE_W - 1) / FILTER_TILE_W, (height + FILTER_TILE_H - 1) / FILTER_TILE_H);

    printf("## 1. Medindo Comunicação (Host -> Device) ##\n");
    //COMUNICAÇÃO H2D
    pthread_t monitor_thread_id;
    thread_data.keep_running = 1;
    thread_data.sample_count = 0;
    pthread_create(&monitor_thread_id, NULL, powerMonitorThread, &thread_data);

    struct timeval time_start_full, time_end_full;
    gettimeofday(&time_start_full, NULL);
    gpuErrchk(cudaEventRecord(start_event));

    gpuErrchk(cudaMemcpy(d_recImage, h_recImage, size_pixels_ushort, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_clip_values, h_clip_values, sizeof(h_clip_values), cudaMemcpyHostToDevice));

    gpuErrchk(cudaEventRecord(stop_event));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gettimeofday(&time_end_full, NULL);

    thread_data.keep_running = 0;
    pthread_join(monitor_thread_id, NULL);

    float time_cuda_ms;
    cudaEventElapsedTime(&time_cuda_ms, start_event, stop_event);
    double time_wall_s = (time_end_full.tv_sec - time_start_full.tv_sec) + (time_end_full.tv_usec - time_start_full.tv_usec) / 1000000.0;

    double total_power_gpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_gpu_mW += thread_data.gpu_power_samples[i];
    double total_power_cpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_cpu_mW += thread_data.cpu_power_samples[i];
    
    double avg_power_W_gpu = (thread_data.sample_count > 0) ? (total_power_gpu_mW / thread_data.sample_count) / 1000.0 : 0.0;
    double avg_power_W_cpu = (thread_data.sample_count > 0) ? (total_power_cpu_mW / thread_data.sample_count) / 1000.0 : 0.0;

    total_comm_h2d_gpu_time_s = time_cuda_ms / 1000.0;
    total_comm_h2d_gpu_energy_j = avg_power_W_gpu * time_wall_s;
    total_comm_h2d_cpu_energy_j = avg_power_W_cpu * time_wall_s;

    printf("    GPU -> Tempo: %.6f s | Potência Média: %6.2f W | Energia: %.4f J\n\n", time_cuda_ms / 1000.0, avg_power_W_gpu, total_comm_h2d_gpu_energy_j);

    printf("## 2. Medindo Execução de Todos os Kernels ##\n");
    // EXECUÇÃO KERNELS
    thread_data.keep_running = 1;
    thread_data.sample_count = 0;
    pthread_create(&monitor_thread_id, NULL, powerMonitorThread, &thread_data);

    gettimeofday(&time_start_full, NULL);
    gpuErrchk(cudaEventRecord(start_event));

    gradientAndClassificationKernel<<<blocksClass, threadsClass>>>(d_recImage, d_classificationMap, d_transformMap, height, width);
    gpuErrchk(cudaPeekAtLastError());

    AlfFilterLuma_kernel<<<blocksFilter, threadsFilter>>>(d_filteredImage, d_recImage, d_classificationMap, d_transformMap, d_filterCoefficients, d_clip_values, width, height, width);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaEventRecord(stop_event));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gettimeofday(&time_end_full, NULL);

    thread_data.keep_running = 0;
    pthread_join(monitor_thread_id, NULL);

    cudaEventElapsedTime(&time_cuda_ms, start_event, stop_event);
    time_wall_s = (time_end_full.tv_sec - time_start_full.tv_sec) + (time_end_full.tv_usec - time_start_full.tv_usec) / 1000000.0;

    total_power_gpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_gpu_mW += thread_data.gpu_power_samples[i];
    total_power_cpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_cpu_mW += thread_data.cpu_power_samples[i];
    avg_power_W_gpu = (thread_data.sample_count > 0) ? (total_power_gpu_mW / thread_data.sample_count) / 1000.0 : 0.0;
    avg_power_W_cpu = (thread_data.sample_count > 0) ? (total_power_cpu_mW / thread_data.sample_count) / 1000.0 : 0.0;

    total_kernel_gpu_time_s = time_cuda_ms / 1000.0;
    total_kernel_gpu_energy_j = avg_power_W_gpu * time_wall_s;
    total_kernel_cpu_energy_j = avg_power_W_cpu * time_wall_s;

    printf("    GPU -> Tempo: %.6f s | Potência Média: %6.2f W | Energia: %.4f J\n\n", time_cuda_ms / 1000.0, avg_power_W_gpu, total_kernel_gpu_energy_j);

    printf("## 3. Medindo Comunicação (Device -> Host) ##\n");
    // COMUNICAÇÃO D2H
    thread_data.keep_running = 1;
    thread_data.sample_count = 0;
    pthread_create(&monitor_thread_id, NULL, powerMonitorThread, &thread_data);

    gettimeofday(&time_start_full, NULL);
    gpuErrchk(cudaEventRecord(start_event));

    gpuErrchk(cudaMemcpy(h_filteredImage, d_filteredImage, size_pixels_ushort, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_classificationMap, d_classificationMap, size_maps, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_transformMap, d_transformMap, size_maps, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventRecord(stop_event));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gettimeofday(&time_end_full, NULL);

    thread_data.keep_running = 0;
    pthread_join(monitor_thread_id, NULL);

    cudaEventElapsedTime(&time_cuda_ms, start_event, stop_event);
    time_wall_s = (time_end_full.tv_sec - time_start_full.tv_sec) + (time_end_full.tv_usec - time_start_full.tv_usec) / 1000000.0;

    total_power_gpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_gpu_mW += thread_data.gpu_power_samples[i];
    total_power_cpu_mW = 0; for(unsigned int i=0; i<thread_data.sample_count; i++) total_power_cpu_mW += thread_data.cpu_power_samples[i];
    avg_power_W_gpu = (thread_data.sample_count > 0) ? (total_power_gpu_mW / thread_data.sample_count) / 1000.0 : 0.0;
    avg_power_W_cpu = (thread_data.sample_count > 0) ? (total_power_cpu_mW / thread_data.sample_count) / 1000.0 : 0.0;

    total_comm_d2h_gpu_time_s = time_cuda_ms / 1000.0;
    total_comm_d2h_gpu_energy_j = avg_power_W_gpu * time_wall_s;
    total_comm_d2h_cpu_energy_j = avg_power_W_cpu * time_wall_s;

    printf("    GPU -> Tempo: %.6f s | Potência Média: %6.2f W | Energia: %.4f J\n", time_cuda_ms / 1000.0, avg_power_W_gpu, total_comm_d2h_gpu_energy_j);

    printf("## ======================= RESUMO GERAL (JETSON) ======================= ##\n");
    printf("  COMUNICACAO TOTAL (H2D + D2H):\n");
    printf("    CPU -> Energia: %.4f J\n", (total_comm_h2d_cpu_energy_j + total_comm_d2h_cpu_energy_j));
    printf("    GPU -> Tempo: %.6f s | Energia: %.4f J\n\n", (total_comm_h2d_gpu_time_s + total_comm_d2h_gpu_time_s), total_comm_h2d_gpu_energy_j + total_comm_d2h_gpu_energy_j);
    
    printf("  EXECUCAO DE TODOS OS KERNELS:\n");
    printf("    CPU -> Energia: %.4f J\n", total_kernel_cpu_energy_j);
    printf("    GPU -> Tempo: %.6f s | Energia: %.4f J\n", total_kernel_gpu_time_s, total_kernel_gpu_energy_j);
    printf("## ===================================================================== ##\n\n");

    printf(" ENERGIA TOTAL GPU: %f J\n",  (total_kernel_gpu_energy_j + total_comm_h2d_gpu_energy_j + total_comm_d2h_gpu_energy_j));
    
    printf("Salvando arquivos de saida...\n");
    saveToCSV("imagem_final_filtrada.csv", h_filteredImage, width, height);
    saveMapToCSV("mapa_de_classes_final.csv", h_classificationMap, width, height);
    saveMapToCSV("mapa_de_transformacao_final.csv", h_transformMap, width, height);
    
    // Limpeza
    cudaFree(d_recImage);
    cudaFree(d_filteredImage);
    cudaFree(d_classificationMap);
    cudaFree(d_transformMap);
    cudaFree(d_filterCoefficients);
    cudaFree(d_clip_values);
    free(h_recImage);
    free(h_filteredImage);
    free(h_classificationMap);
    free(h_transformMap);
    free(thread_data.gpu_power_samples);
    free(thread_data.cpu_power_samples);
    
    printf("Processo concluido com sucesso.\n");
    return 0;
}
