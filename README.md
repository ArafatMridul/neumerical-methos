# neumerical-methos

# Non-Linear Equations
An equation whose variable is raised to a power greater than one or is involved in functions like trigonometric, exponential.

## Bisection Method
A numerical technique to find a root of a continuous function.

### Steps in the Bisection Method:
- **Step 1:** Choose two initial points, \( a \) and \( b \), such that \( f(a) \) and \( f(b) \) must have two opposite signs.
- **Step 2:** Calculate the midpoint \( c = \frac{a + b}{2} \).
- **Step 3:** Evaluate \( f(c) \):
  - If \( f(c) = 0 \), then \( c \) is the root.
  - If \( f(a) \) and \( f(c) \) have opposite signs, then set \( b = c \) (the root is in the interval \([a, c]\)).
  - Otherwise then set \( a = c \) (the root is in the interval \([c, b]\)).
- **Step 4:** Repeat Steps 2 and 3 until the desired accuracy is achieved; it is defined here.



# False Position Method
  - A numerical technique used to find roots of a continuous function. 
  
## Steps in False Position Method

1. *Input*:
   - Define the function \(f(x)\).
   - Choose initial guesses \(a\) and \(b\) such that \(f(a) \cdot f(b) < 0\) .
   - Set a tolerance level \(\text{tolerance}\) and a maximum number of iterations \

2. *Check Initial Conditions*:
   - If \(f(a) \cdot f(b) \geq 0\):
     - Show an error message and exit

3. *Iteration*:
   - For \(i\) from 1 to \(\text{max iterations}\):
     1. Calculate the point of intersection (the approximation of the root):
        \[
        c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}
        \]
     2. Evaluate \(f(c)\).
     3. Check if \(|f(c)| < \text{tolerance}\):
        - If true, print the root \(c\) and the number of iterations taken, then exit.
     4. Then update the interval:
        - If \(f(a) \cdot f(c) < 0\),then set \(b = c\) (the root is in the left subinterval).
        - Otherwise set \(a = c\) (the root is in the right subinterval).

1. *Output*:
   - If the maximum number of iterations is reached without finding a root within the tolerance, return the best approximation of the root.This is the most optimal solution here.



## Secant Method



### Algorithm Steps

1. *Input*:
   - Define the function \(f(x)\) as a string.
   - Choose two initial guesses \(x_0\) and \(x_1\).
   - Set a tolerance level \(\text{tolerance}\) and a maximum number of iterations \(\text{maxIterations}\).

2. *Evaluate Function*:
   - Calculate \(f_0 = f(x_0)\) and \(f_1 = f(x_1)\).

3. *Iteration*:
   - For \(i\) from 0 to \(\text{maxIterations} - 1\):
     1. Check if the function values are too close (if \(|f_1 - f_0| < 1e-10\)):
        - If true, print an error message and stop the iteration.
     2. Calculate the next approximation using the formula:
        \[
        x_2 = x_1 - f_1 \cdot \frac{x_1 - x_0}{f_1 - f_0}
        \]
     3. Evaluate \(f_2 = f(x_2)\).
     4. Check for convergence:
        - If \(|x_2 - x_1| < \text{tolerance}\), return \(x_2\) as the root.
     5. Update values:
        - Set \(x_0 = x_1\), \(f_0 = f_1\), \(x_1 = x_2\), \(f_1 = f_2\).

4. *Output*:
   - If the maximum iterations are reached without converging, print a message and return the last approximation \(x_1\).

## Newton-Raphson Method

### Algorithm Steps

1. *Input*:
   - Define the function \(f(x)\) as a string.
   - Choose an initial guess \(x\).
   - Set a tolerance level \(\text{tolerance}\) and a maximum number of iterations \(\text{maxIterations}\).

2. *Iteration*:
   - For \(i\) from 0 to \(\text{maxIterations} - 1\):
     1. Evaluate \(f(x)\) and its derivative \(f'(x)\).
     2. Check if the derivative is too small (if \(|f'(x)| < 1e-10\)):
        - If true, print an error message and stop the iteration.
     3. Calculate the new approximation using the formula:
        \[
        x_{\text{new}} = x - \frac{f(x)}{f'(x)}
        \]
     4. Check for convergence:
        - If \(|x_{\text{new}} - x| < \text{tolerance}\), return \(x_{\text{new}}\) as the root.
     5. Update \(x\) to \(x_{\text{new}}\).

3. *Output*:
   - If the maximum iterations are reached without converging, print a message and return the last approximation \(x\).




