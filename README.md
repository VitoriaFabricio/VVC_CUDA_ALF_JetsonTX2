# CUDA VVC Adaptive Loop Filter (ALF) for Jetson TX2

This repository contains an optimized implementation of the **Adaptive Loop Filter (ALF)** for **NVIDIA Jetson TX2** embedded systems. It implements the luma filter defined in the **VVC (Versatile Video Coding / H.266)** standard using CUDA.

Unlike the desktop version, this implementation reads power metrics directly from the **INA3221** hardware sensors via `sysfs`, removing the dependency on the NVML library which is not supported on Jetson architectures.

## 🚀 Features

* **Embedded Optimization:** Tailored for Jetson TX2, monitoring CPU and GPU power rails directly from hardware sensors.
* **10-bit Support:** Uses `unsigned short` to correctly process high bit-depth video (0-1023), preventing integer overflow artifacts.
* **Block Classification:** Gradient-based classification (Laplacian) on 4x4 blocks with an 8x8 **overlapping window**.
* **Geometric Transformation:** Applies rotation and mirroring to align texture direction with filter coefficients.
* **Luma Filtering:** 7x7 diamond-shaped Wiener filter application.

## 🛠️ Prerequisites

* **Hardware:** NVIDIA Jetson TX2 (Default configuration).
    * *Note: Can be adapted for Jetson Nano, Xavier, or Orin by changing sensor paths.*
* **Software:** JetPack SDK (includes CUDA Toolkit).
* **Permissions:** Root/Sudo access (required to read `/sys/bus/i2c/...` power files).

## ⚙️ Configuration

### 1. Image Resolution & Bit Depth
To change the input resolution or bit depth, edit the constants at the beginning of `main.cu`:

```c
const int width = 1920;  // Image width
const int height = 1080; // Image height
// ...
#define BIT_DEPTH 10     // Set to 8, 10, or 12
```

### 2. Power Sensor Paths (Crucial for Jetson)
The code is hardcoded for the **Jetson TX2** power rails. If you are using a different Jetson board (e.g., Nano, Xavier AGX, Orin), you must update the paths in the `powerMonitorThread` function (approx. line 270):

```c
// Default paths for Jetson TX2
const char* gpu_power_path = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input";
const char* cpu_power_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input";
```

**How to find paths for your board:**
Run the following command on your Jetson to list available power sensors:
```bash
find /sys/bus/i2c/drivers/ina3221x -name "in_power*_input"
```
*You will need to identify which file corresponds to the GPU and CPU rails for your specific hardware guide.*

## 📦 Compilation

Do **not** link against `nvidia-ml`. Use the following command:

```bash
nvcc main.cu -o alf_jetson -lpthread
```

## ▶️ How to Use

1.  **Prepare the Input File:**
    Place a file named **`original_0.csv`** in the same directory.
    * **Format:** Integer pixel values (Luma) separated by commas or spaces.
    * **Size:** Must contain exactly `width * height` values.

2.  **Run the Program (Requires Sudo):**
    Because the code reads system files for power monitoring, you must run it with superuser privileges:

    ```bash
    sudo ./alf_jetson
    ```

## 📂 Output Files

The program generates three CSV files for analysis:

1.  **`imagem_final_filtrada.csv`**: The filtered image pixels (10-bit range).
2.  **`mapa_de_classes_final.csv`**: The Class ID (0-24) assigned to each 4x4 block.
3.  **`mapa_de_transformacao_final.csv`**: The Geometric Transformation ID (0-3) applied to each block.

## 📝 Technical Details

* **Shared Memory Strategy:** Uses a Gather strategy to handle the VVC 8x8 overlapping window requirement efficiently within the classification kernel.
* **Direct Sysfs Reading:** Bypasses high-level APIs to read the INA3221 voltage/current monitor directly, providing accurate power consumption data for the embedded platform.
