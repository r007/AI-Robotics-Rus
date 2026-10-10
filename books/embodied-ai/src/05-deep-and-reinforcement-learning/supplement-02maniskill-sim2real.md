# Дополнение 02. ManiSkill: перенос из симуляции в реальность

Этот раздел довольно краткий: в основном нужно выполнить команды. Если вам интересно, можно повторить такой эксперимент:

```bash
mamba create -n py311 "python==3.11"   # рекомендуется Python 3.11
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
# Если torch установился без поддержки CUDA, используйте следующую команду
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/   # установите версию torch, которая подходит для вашей системы
```

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# Код основан на чуть более старом коммите lerobot. Недавно LeRobot изменил расположение некоторых файлов, которые мы импортируем, поэтому часть импортов сломалась.
# Актуальная версия LeRobot тоже работает, но пути импорта нужно обновить.
git reset --hard a989c795587d122299275c65a38ffdd0a804b8dc
pip install -e .
```

Далее настройте Vulkan, как описано в разделе «Вспомогательные инструменты» (`other tools`) этой книги.

```bash
python -m mani_skill.examples.demo_random_action
```

Дальнейшие инструкции см. в репозитории:

https://github.com/StoneT2000/lerobot-sim2real.git
