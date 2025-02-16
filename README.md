
# Методы оптимизации: алгоритм Гаусса–Зейделя

Данный проект демонстрирует пример реализации алгоритма циклического покоординатного метода Гаусса–Зейделя для решения задачи многомерной безусловной оптимизации. Код написан на языке Python и использует библиотеки **NumPy** и **Matplotlib**.

## Описание

- **Алгоритм Гаусса–Зейделя** в данном проекте реализован как покоординатный метод оптимизации.  
- На каждом шаге выполняется спуск по одной координате (x1, x2, …), пока не будет достигнуто заданное условие останова (по модулю изменения вектора решения).  
- В качестве примера можно оптимизировать функцию вида \((x1 - 4*x2)^2 + (x2 + 5)^2\), однако при желании функцию можно задать в любом другом аналитическом виде, используя синтаксис Python (x1, x2).

## Основные файлы

- **main.py** (условно): содержит код с определением всех функций и реализацией основного алгоритма.
- **requirements.txt** (при необходимости): список зависимостей (библиотек), требуемых для работы программы, например:
  ```txt
  numpy
  matplotlib
  ```
- **README.md**: описание проекта и инструкции по запуску (именно этот файл).

## Установка и запуск

1. **Клонируйте** репозиторий или скачайте архив с проектом:
   ```bash
   git clone https://github.com/IlyaKudrya2000/gauss-seidel-optimization
   ```
2. **Перейдите** в папку проекта:
   ```bash
   cd <название_репозитория>
   ```
3. **Установите** необходимые зависимости (при наличии файла *requirements.txt*):
   ```bash
   pip install -r requirements.txt
   ```
   Иначе отдельно:
   ```bash
   pip install numpy matplotlib
   ```
4. **Запустите** основной скрипт:
   ```bash
   python main.py
   ```
5. **Следуйте** инструкциям, которые будут выведены в консоли:
   - Введите выражение функции (например, `(x1 - 4*x2)**2 + (x2 + 5)**2`)
   - Укажите начальные приближения для x1 и x2
   - Задайте значение точности (eps), например `0.001`

## Пример запуска

```text
Введите функцию вида (x1 - 4*x2)**2 + (x2 + 5)**2: (x1 - 4*x2)**2 + (x2 + 5)**2
Введите начальное значение x1: 10
Введите начальное значение x2: -5
Введите желаемую точность, например 0.01: 0.001
```

После чего в консоли появится список итераций, значения функции на каждом шаге, а также критерий остановки. Завершится алгоритм сообщением о найденном минимуме и построит график изменения функции по итерациям.

## Пример результата

- Пошаговый вывод (x1, x2, значение функции, норма изменения, критерий достижения).
- График, который отобразится в отдельном окне при помощи **Matplotlib**.
