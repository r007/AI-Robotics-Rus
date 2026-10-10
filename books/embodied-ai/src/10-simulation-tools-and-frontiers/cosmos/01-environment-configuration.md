```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict1.git
cd cosmos-predict1



# Создаём conda-окружение cosmos-predict1.
conda env create --file cosmos-predict1.yaml
# Активируем conda-окружение cosmos-predict1.
conda activate cosmos-predict1
# Устанавливаем зависимости.
pip install -r requirements.txt
# Исправляем проблемы линковки Transformer Engine в conda-окружениях.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Устанавливаем Transformer Engine.
pip install transformer-engine[pytorch]==1.12.0
```
