# Важная настройка окружения Vulkan

### Vulkan

### Ubuntu

**Установка Vulkan в Ubuntu:**

**Bash**

```bash
sudo apt-get install libvulkan1
```

**Проверка установки Vulkan:**

**Bash**

```bash
sudo apt-get install vulkan-tools
vulkaninfo
```

Если `vulkaninfo` не показывает сведений о Vulkan, проверьте наличие следующих файлов:

```
/usr/share/vulkan/icd.d/nvidia_icd.json
/usr/share/glvnd/egl_vendor.d/10_nvidia.json
/etc/vulkan/implicit_layer.d/nvidia_layers.json (необязательно, но для некоторых GPU, например A100, нужно)
```

Если файла `/usr/share/vulkan/icd.d/nvidia_icd.json` нет, создайте его со следующим содержимым:

**JSON**

```json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
```

Если файла `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` нет, попробуйте установить `sudo apt-get install libglvnd-dev`. Файл `10_nvidia.json` содержит:

**JSON**

```json
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
```

Если файла `/etc/vulkan/implicit_layer.d/nvidia_layers.json` нет, создайте его со следующим содержимым:

**JSON**

```json
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
```

Обсуждение этой проблемы есть в [issue #52 репозитория ManiSkill](https://github.com/haosulab/ManiSkill/issues/52).

Если драйвер Vulkan повреждён, могут появляться следующие ошибки. Попробуйте переустановить драйвер по инструкции выше.

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
```

```
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
```

```
Segmentation fault (core dumped)
```

Если проблемы остаются, проверьте драйверы NVIDIA. Сначала выполните:

**Bash**

```bash
ldconfig -p | grep libGLX_nvidia
```

Если `libGLX_nvidia.so` не найден, скорее всего, установлен неподходящий драйвер. В Linux правильный драйвер рекомендуется ставить пакетом **`nvidia-driver-xxx`** (не используйте версии с `server` в имени пакета) и не использовать другие способы, например установщик `runfile`.

Для WSL:

```bash
sudo -E apt-get install -y libnvidia-gl-580
```

Флаг `-E` сохраняет переменные окружения, в том числе сетевые настройки прокси.

```bash
ldconfig -p | grep libGLX_nvidia
        libGLX_nvidia.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libGLX_nvidia.so.0
```

*Примечание для WSL.* В официальной документации написано (по-английски): *If you still have some issues, you can check the NVIDIA drivers. First run `ldconfig -p | grep libGLX_nvidia`. If libGLX_nvidia.so is not found, then it is likely that you have installed an incorrect driver. To get the right driver on linux, it is recommended to install nvidia-driver-xxx (do not use the ones with server in the package name) and to avoid using any other method of installation like a runfile.* Это описание для обычного Linux. В WSL драйвер NVIDIA не устанавливается внутри Linux: `nvidia-smi` работает, потому что драйвер установлен в Windows, а `nvcc` без CUDA Toolkit недоступен. Поэтому нужны и драйвер, и Toolkit, но устанавливать их надо по инструкциям для WSL, а не для bare-metal Linux.

В WSL легко перепутать две части:

1. **Драйвер в Windows.** Команда `nvidia-smi` работает потому, что драйвер NVIDIA установлен в Windows. WSL2 использует специальный механизм и обращается к этому драйверу напрямую, но предоставляет только базовый мост.
2. **Toolkit и пользовательские библиотеки в Linux.** Инструменты вроде `nvcc` (компилятор CUDA) и более глубокие библиотеки, которые нужны SAPIEN и Vulkan (например, `libcuda.so`, `libvulkan.so` и файл `nvidia_icd.json`, найденный выше), сами по себе в WSL не появляются. Их нужно установить внутри WSL через CUDA Toolkit for WSL.

Инструкция из документации относится к Linux на физическом компьютере (bare-metal), где драйвер ставится через `apt install nvidia-driver-xxx`. В WSL так делать нельзя: это может нарушить связь между драйвером WSL и драйвером Windows.

Итак, в WSL (Ubuntu) нужно установить официальный NVIDIA CUDA Toolkit. Это решает и проблему с `nvcc`, и проблему совместимости драйвера Vulkan. См. раздел «Установка драйвера CUDA и nvcc в WSL» в этом разделе.
