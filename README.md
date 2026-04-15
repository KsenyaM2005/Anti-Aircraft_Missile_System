# Проект ЗРС

Симулятор зенитно-ракетной системы с тактовым обновлением компонентов:

- воздушная обстановка
- радиолокаторы
- пункт боевого управления
- пусковые установки
- зенитные управляемые ракеты
- графический интерфейс
- диспетчер симуляции

Проект моделирует обнаружение целей, формирование треков, принятие решения на поражение, пуск ракеты и отображение всей обстановки в GUI.

## Быстрый старт

Установить зависимости:

```powershell
pip install -r requirements.txt
```

Запуск GUI одной командой:

```powershell
.\start_project.bat
```

То же напрямую через Python:

```powershell
python start_project.py
```

## Режимы запуска

GUI:

```powershell
.\start_project.bat
```

Headless:

```powershell
.\start_project.bat headless --duration 5
```

Быстрый консольный smoke test:

```powershell
.\start_project.bat quick
```

Автотесты:

```powershell
.\start_project.bat test
```

Полная проверка проекта:

```powershell
.\start_project.bat check
```

Прямой вызов через Python поддерживает те же режимы:

```powershell
python start_project.py gui
python start_project.py headless --duration 5
python start_project.py quick
python start_project.py test
python start_project.py check
```

## Управление в GUI

- `Space` — пауза / продолжить
- `R` — сбросить вид
- `1` — вид сверху
- `2` — боковой вид
- `3` — split view
- `F` — следовать за выбранной целью
- `H` — показать / скрыть HUD
- `Esc` — снять выделение
- колесо мыши — масштаб
- средняя кнопка мыши — панорамирование

## Структура проекта

- [start_project.py](./start_project.py) — основной entrypoint проекта
- [start_project.bat](./start_project.bat) — простой запуск для Windows
- [config.yaml](./config.yaml) — конфигурация среды, РЛС, ПБУ и ПУ
- [src/main.py](./src/main.py) — основной симулятор
- [src/dispatcher.py](./src/dispatcher.py) — диспетчер тактов и склейка компонентов
- [src/air_environment.py](./src/air_environment.py) — воздушная обстановка
- [src/radar.py](./src/radar.py) — радиолокатор и треки
- [src/pbu.py](./src/pbu.py) — логика ПБУ
- [src/launcher.py](./src/launcher.py) — логика пусковой установки
- [src/projectile.py](./src/projectile.py) — логика ЗУР
- [src/gui.py](./src/gui.py) — графический интерфейс
- [test](./test) — автотесты

## Как устроен обмен данными

Упрощённо цикл работы выглядит так:

1. Диспетчер выдаёт очередной такт времени.
2. Воздушная обстановка обновляет истинные координаты целей и ракет.
3. РЛС получает зашумленные измерения и формирует треки.
4. ПБУ оценивает угрозы и принимает решение на сопровождение и пуск.
5. ПУ получает команду, готовится и выполняет пуск.
6. ЗУР летит, получает команды наведения и пытается перехватить цель.
7. GUI отображает текущую обстановку и принимает команды оператора.

## Проверка, что всё работает

Минимальная рекомендуемая проверка:

```powershell
.\start_project.bat check
```

Эта команда выполняет:

- автотесты
- быстрый консольный прогон
- headless запуск
- GUI smoke test без ручного открытия окна

## Примечания

- Проект ориентирован на запуск из корня репозитория.
- Основная конфигурация лежит в `config.yaml`.
- Логи сохраняются в папку `logs`.
