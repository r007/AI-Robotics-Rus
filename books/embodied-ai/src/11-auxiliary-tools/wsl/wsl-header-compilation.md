# Сборка заголовков ядра для WSL

В WSL иногда нужно собрать заголовки (headers) ядра. Проверить версию текущего ядра можно так:

```bash
$ cat /proc/version
Linux version 6.6.87.2-microsoft-standard-WSL2 (root@439a258ad544) (gcc (GCC) 11.2.0, GNU ld (GNU Binutils) 2.37) #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025
Linux version 6.6.87.2-microsoft-standard-WSL2 (xxx@xxx) (gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #1 SMP PREEMPT_DYNAMIC Fri Oct  3 18:14:58 CST 2025
```

Сборка ядра WSL2 из исходников:

```bash
wget https://github.com/microsoft/WSL2-Linux-Kernel/archive/refs/tags/linux-msft-wsl-6.6.87.2.tar.gz
tar -zxvf linux-msft-wsl-6.6.87.2.tar.gz
cd WSL2-Linux-Kernel-linux-msft-wsl-6.6.87.2/
sudo apt-get update
sudo apt install build-essential flex bison dwarves libssl-dev libelf-dev cpio qemu-utils
make menuconfig KCONFIG_CONFIG=Microsoft/config-wsl
make -j$(nproc) KCONFIG_CONFIG=Microsoft/config-wsl
make -j$(nproc) INSTALL_MOD_PATH="$PWD/modules" modules_install
sudo ./Microsoft/scripts/gen_modules_vhdx.sh "$PWD/modules" $(make -s kernelrelease) modules.vhdx
```

Результат:

- файл модулей `modules.vhdx` находится в `WSL2-Linux-Kernel-linux-msft-wsl-6.6.87.2/modules.vhdx`;
- собранное ядро находится в `WSL2-Linux-Kernel-linux-msft-wsl-6.6.87.2/arch/x86/boot/bzImage`.
