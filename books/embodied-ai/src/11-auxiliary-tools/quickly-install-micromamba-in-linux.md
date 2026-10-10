# Быстрая установка micromamba в Linux

```bash
sudo apt install bzip2

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# ./bin/micromamba shell init -s bash -p /opt/micromamba
# В последних версиях параметр -p больше не поддерживается

./bin/micromamba shell init -s bash --root-prefix ./micromamba
```

Откройте `~/.bashrc` и добавьте:

```bash
alias mamba=micromamba
```

Затем настройте файл `.condarc`.
