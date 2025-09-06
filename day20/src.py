import numpy as np
import matplotlib.pyplot as plt
import time
plt.rc('font', family='Apple SD Gothic Neo')

# ** 기초 수학 **
# p.128 수치적 미분
def numerical_derivative(f, x, h=1e-7):
    return (f(x+h) - f(x)) / h

def derivate_definition_demo():
    def f(x):
        return x**2
    
    a = 2
    h_values = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    for h in h_values:
        diff_quotient = numerical_derivative(f, a, h)
        print(f"h = {h:8.5f}: [f(2+{h}) - f(2)] / {h} = {diff_quotient:.6f}")
    
    analytical_derivative = 2 * a
    print(f"\n정확한 미분값 : {analytical_derivative}")
    numerical_result = numerical_derivative(f, a)
    print(f"수치적 미분값 : {numerical_result:.6f}")

    return h_values, [f(a+h) for h in h_values]

# 미분 시각화 (할선 -> 접선 변화 시각화)
def plot_derivative_concept():
    def f(x):
        return x**2
    
    x = np.linspace(0, 4, 100)
    y = f(x)

    a = 2
    slope = 2*a

    plt.figure(figsize=(10,6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x^2')

    plt.plot(a, f(a), 'ro', markersize= 8, label=f'점 {a}, {f(a)})')

    tangent_x = np.linspace(1, 3, 100)
    tnagent_y = f(a) + slope * (tangent_x-a)

    plt.plot(tangent_x, tnagent_y, 'r--', linewidth = 2, 
             label=f'접선 (기울기 = {slope})')
    
    h_values = [1, 0.5, 0.2]
    colors = ['orange', 'green', 'purple']

    for i, h in enumerate(h_values):
        x_end = a + h
        y_end = f(x_end)
        secant_slope = numerical_derivative(f, a, h)
        secant_x = np.array([a, x_end])
        secant_y = np.array([f(a), y_end])
        plt.plot(secant_x, secant_y, '--', color=colors[i],
                 linewidth=1.5, label=f"할선 (h={h}, 기울기={secant_slope:.1f})")
        plt.plot(x_end, y_end, 'o', color=colors[i], markersize=6)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('미분의 기하학적 의미 : 할선이 접선으로 수렴')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_derivative_concept()
  
# p.149 편미분
class PartialDerivatives:
    @staticmethod
    def numerical_partial_derivative(f, x, y, var='x', h=1e-7):
        if var == 'x':
            return (f(x + h, y) - f(x - h, y)) / (2 * h)
        elif var == 'y':
            return (f(x, y + h) - f(x, y - h)) / (2 * h)
    
    @staticmethod
    def gradient(f, x, y, h=1e-7):
        df_dx = PartialDerivatives.numerical_partial_derivative(f, x, y ,'x', h)
        df_dy = PartialDerivatives.numerical_partial_derivative(f, x, y ,'y', h)

        return np.array([df_dx, df_dy])

def partial_derivative_examples():
    def f(x, y):
        return x**2 * y + 3*x * y**2 + 5*x + 2*y + 7

    def df_dx_analytical(x, y):
        return 2*x*y + 3*y**2 + 5

    def df_dy_analytical(x, y):
        return x**2 + 6*x*y + 2

    x, y = 2, 3

    analytical_x = df_dx_analytical(x, y)
    print(f" 해석적 : {analytical_x}")
    numerical_x = PartialDerivatives.numerical_partial_derivative(f, x, y, 'x')
    print(f" 수학적 : {numerical_x:.6f}")
    print(f" 오차 : {abs(analytical_x - numerical_x):.2e}")

    analytical_y = df_dy_analytical(x, y)
    print(f" 해석적 : {analytical_y}")
    numerical_y = PartialDerivatives.numerical_partial_derivative(f, x, y, 'y')
    print(f" 수학적 : {numerical_y:.6f}")
    print(f" 오차 : {abs(analytical_y - numerical_y):.2e}")

    grad_analytical = np.array([analytical_x, analytical_y])
    grad_numerical = PartialDerivatives.gradient(f, x, y)
    print(f" 해석적 : [{analytical_x}, {analytical_y}]")
    print(f" 수학적 : [{grad_numerical[0]:.6f}, {grad_numerical[1]:.6f}]")
    print(f" 크기 : {np.linalg.norm(grad_analytical):.6f}")

# 경사 하강법
def objective_function(x):
    return (x-5)**2 + 3

def gradient(x):
    return 2*(x-5)

x = 0
learning_rate = 0.1
iterations = 20

for i in range(iterations):
    current_value = objective_function(x)
    current_gradient = gradient(x)
    print(f"반복 {i+1:2d}: x = {x:7.4f}, f(x) = {current_value:7.4f}, 경사도 = {current_gradient:7.4f}")

    x = x - learning_rate * current_gradient
    if abs(current_gradient) < 1e-6:
        print("수렴!")
        break
    print(f"\n최종 결과 : x = {x:.6f}, f(x) = {objective_function(x):.6f}")
