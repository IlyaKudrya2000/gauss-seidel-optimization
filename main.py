import numpy as np
import matplotlib.pyplot as plt
import re

def build_function(user_expr):
    """
    Преобразует введенное пользователем выражение 
    в функцию от x1 и x2 с помощью eval.
    """
    def target_fun(x1, x2):
        temp_expr = re.sub(r'\bx1\b', f'({x1})', user_expr)
        temp_expr = re.sub(r'\bx2\b', f'({x2})', temp_expr)
        return eval(temp_expr)
    return target_fun

def coordinate_descent_1d(coords, axis_index, lr=0.01, max_cycles=100, func=None):
    """
    Ищет минимум по одной из координат (axis_index) 
    методом псевдоградientного шага при фиксированных других координатах.
    """
    updated_coords = np.array(coords, dtype=float)
    for _ in range(max_cycles):
        # Приращение для вычисления 'производной' по нужной координате
        if axis_index == 0:
            derivative = (func(updated_coords[0] + 1e-5, updated_coords[1]) 
                          - func(updated_coords[0], updated_coords[1])) / 1e-5
        else:
            derivative = (func(updated_coords[0], updated_coords[1] + 1e-5) 
                          - func(updated_coords[0], updated_coords[1])) / 1e-5

        # Обновляем значение соответствующей координаты
        updated_coords[axis_index] -= lr * derivative
        
        # Если 'производная' стала достаточно маленькой, завершаем итерации
        if abs(derivative) < 1e-4:
            break

    return updated_coords[axis_index]

def gauss_zeidel_solver(func, start_coords, eps):
    """
    Главная процедура для минимизации функции методом 
    покоординатного Гаусса–Зейделя.
    """
    coords = np.array(start_coords, dtype=float)
    log_list = []
    limit_iter = 1000

    for step in range(limit_iter):
        old_coords = coords.copy()

        # Сначала спуск по x1
        coords[0] = coordinate_descent_1d(coords, axis_index=0, func=func)
        # Затем по x2
        coords[1] = coordinate_descent_1d(coords, axis_index=1, func=func)

        current_val = func(coords[0], coords[1])
        shift_norm = np.linalg.norm(coords - old_coords)
        stop_crit = "достигнут" if shift_norm < eps else "не достигнут"

        # Сохраняем промежуточные результаты
        log_list.append([
            old_coords[0],
            old_coords[1],
            current_val,
            round(shift_norm, 4),
            stop_crit
        ])

        # Проверяем условие завершения
        if shift_norm < eps:
            break

    return log_list, coords

# Получение данных из консоли
user_expr = input("Введите функцию вида (x1 - 4*x2)**2 + (x2 + 5)**2: ")
target_fun = build_function(user_expr)

start_x1 = float(input("Введите начальное значение x1: "))
start_x2 = float(input("Введите начальное значение x2: "))
accuracy = float(input("Введите желаемую точность, например 0.01: "))

# Запуск алгоритма
iteration_data, final_coords = gauss_zeidel_solver(
    func=target_fun, 
    start_coords=[start_x1, start_x2], 
    eps=accuracy
)

# Печать результатов по итерациям
print("\nПошаговые данные оптимизации:")
for idx, item in enumerate(iteration_data):
    print(f"Итерация {idx + 1}: "
          f"x1 = {item[0]:.4f}, "
          f"x2 = {item[1]:.4f}, "
          f"f(x) = {item[2]:.6f}, "
          f"||Δx|| = {item[3]}, "
          f"критерий: {item[4]}")

print(f"\nМинимум найден в точке: x1 = {final_coords[0]:.4f}, x2 = {final_coords[1]:.4f}")

# Визуализация процесса сходимости
function_values = [row[2] for row in iteration_data]
plt.plot(range(1, len(function_values) + 1), function_values, marker='o')
plt.title("Динамика сходимости метода Гаусса–Зейделя")
plt.xlabel("Номер итерации")
plt.ylabel("Значение целевой функции f(x)")
plt.grid(True)
plt.show()
