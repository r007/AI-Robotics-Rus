# Установка китайского метода ввода в Ubuntu

```bash
sudo apt update
sudo apt install ibus-libpinyin
ibus restart

ibus-setup
```

Команда `ibus-setup` открывает окно «IBus Preferences». В нём:

1. Откройте вкладку **Input Method** (Метод ввода).
2. Нажмите **Add** (Добавить) справа.
3. В списке последовательно выберите **Chinese** (Китайский) → **Intelligent Pinyin** (Интеллектуальный пиньинь) → **Add** (Добавить).
