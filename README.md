# neumerical-methos

# Linear Equations
Linear equations are algebraic equations where each term is either a constant or the product of a constant and a single variable. They graph as straight lines on a coordinate plane. 

## Jacobi Iterative Method

The Jacobi Iterative Method is an algorithm for solving systems of linear equations. It’s useful for large systems where direct methods, like Gaussian elimination, become impractical. This method iteratively updates each variable based on the values from the previous iteration.

## Problem Setup

We aim to solve a system of linear equations represented by:

\[
Ax = b
\]

where:
- \( A \) is a square matrix (coefficients of the system),
- \( x \) is the vector of unknowns,
- \( b \) is the result vector.

The Jacobi method is particularly effective if \( A \) is diagonally dominant or if the system is otherwise suitable for convergence.

## Steps of the Jacobi Iterative Method

1. **Rewrite Equations**  
   Rewrite each equation to isolate \( x_i \) in terms of other variables:

   \[
   a_{ii} x_i = b_i - \sum_{j \neq i} a_{ij} x_j
   \]

2. **Initial Guess**  
   Choose an initial guess \( x^{(0)} \) for the solution vector \( x \). Often, \( x^{(0)} = 0 \) is used.

3. **Iterate**  
   For each iteration \( k \), update each \( x_i \) as:

   \[
   x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right)
   \]

   **Note**: Use values from the previous iteration, \( x^{(k)} \), not from the current iteration.

4. **Check Convergence**  
   After each iteration, check if the solution has converged by evaluating:

   \[
   ||x^{(k+1)} - x^{(k)}|| < \epsilon
   \]

   where \( \epsilon \) is a small positive tolerance level. If this condition is met, the method has converged, and \( x^{(k+1)} \) is an approximate solution. Otherwise, repeat the iteration.

5. **Repeat Until Convergence**

## Advantages
- **Parallelizable**: Each variable can be updated independently.
- **Simplicity**: Easy to implement.

## Limitations
- **Convergence**: Requires the matrix \( A \) to be diagonally dominant or the method may not converge.
- **Slow**: May require many iterations for an accurate solution.

## Usage

To use the Jacobi method:
1. Define the system of equations.
2. Implement the iteration steps.
3. Set a convergence tolerance \( \epsilon \).
4. Iterate until convergence criteria are met.

## Gauss-Seidel Iterative Method

The Gauss-Seidel Iterative Method is an algorithm for solving systems of linear equations, especially effective for large systems or sparse matrices where direct methods are computationally expensive. It improves upon the Jacobi method by using the most recently updated values during each iteration.

## Problem Setup

We aim to solve a system of linear equations represented by:

\[
Ax = b
\]

where:
- \( A \) is a square matrix of coefficients,
- \( x \) is the vector of unknowns,
- \( b \) is the result vector.

The Gauss-Seidel method is most effective when \( A \) is a diagonally dominant matrix, which often ensures convergence.

## Steps of the Gauss-Seidel Iterative Method

1. **Rewrite Equations**  
   Rearrange each equation so that each variable \( x_i \) is isolated. For each equation:

   \[
   a_{ii} x_i = b_i - \sum_{j \neq i} a_{ij} x_j
   \]

2. **Initial Guess**  
   Choose an initial guess \( x^{(0)} \) for the solution vector \( x \). A common choice is \( x^{(0)} = 0 \).

3. **Iterate**  
   For each iteration \( k \), update each variable \( x_i \) sequentially using:

   \[
   x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)}\right)
   \]

   - **Note**: Unlike the Jacobi method, the Gauss-Seidel method updates each \( x_i \) immediately using the latest available values.

4. **Check Convergence**  
   After each iteration, check if the solution has converged by evaluating:

   \[
   ||x^{(k+1)} - x^{(k)}|| < \epsilon
   \]

   where \( \epsilon \) is a small positive tolerance level. If this condition is met, the method has converged, and \( x^{(k+1)} \) is an approximate solution. If not, continue iterating.

5. **Repeat Until Convergence**

## Advantages
- **Faster Convergence**: Generally converges more quickly than the Jacobi method by using updated values immediately.
- **Simplicity**: Straightforward to implement for large, sparse systems.

## Limitations
- **Convergence**: Requires a diagonally dominant or symmetric positive-definite matrix for guaranteed convergence.
- **Sequential Dependence**: Not as parallelizable as the Jacobi method, as each update depends on the previous one.

## Usage

To use the Gauss-Seidel method:
1. Define the system of equations.
2. Implement the iteration steps, updating each variable sequentially.
3. Set a convergence tolerance \( \epsilon \).
4. Iterate until the convergence criteria are met.


# Gauss Elimination Method

The Gauss Elimination Method is a systematic, non-iterative algorithm for solving systems of linear equations. It transforms a given system into an upper triangular form, allowing for straightforward back-substitution to find the solution.

## Problem Setup

We aim to solve a system of linear equations represented by:

\[
Ax = b
\]

where:
- \( A \) is a square matrix of coefficients,
- \( x \) is the vector of unknowns,
- \( b \) is the result vector.

## Steps of the Gauss Elimination Method

The method has two main phases: **Forward Elimination** and **Back Substitution**.

### 1. Forward Elimination

Transform the matrix \( A \) into an upper triangular matrix \( U \) by eliminating entries below the main diagonal.

1. **Pivot Selection**  
   For each row \( i \), select a pivot element \( a_{ii} \) (typically the largest absolute value in the column) to reduce computational error.

2. **Row Operations**  
   For each subsequent row \( j > i \), eliminate the entry \( a_{ji} \) by using the row operation:

   \[
   \text{Row } j = \text{Row } j - \left(\frac{a_{ji}}{a_{ii}}\right) \times \text{Row } i
   \]

   This process zeros out entries below the main diagonal, transforming \( A \) into an upper triangular matrix \( U \).

### 2. Back Substitution

Once the matrix \( A \) is in upper triangular form, solve for each variable starting from the last row.

1. For the last row \( n \):
   
   \[
   x_n = \frac{b_n}{a_{nn}}
   \]

2. Substitute \( x_n \) into the previous row to solve for \( x_{n-1} \), then proceed upward:

   \[
   x_i = \frac{b_i - \sum_{j=i+1}^{n} a_{ij} x_j}{a_{ii}}
   \]

Continue this process until all variables are solved.

### Step-by-Step Solution

1. **Forward Elimination**:
   - Use the first equation to eliminate \( x \) terms from the second and third equations.
   - Use the modified second equation to eliminate \( y \) terms from the third equation.

2. **Back Substitution**:
   - Start from the last equation to solve for \( z \).
   - Substitute \( z \) into the second equation to solve for \( y \), then use both values to solve for \( x \) in the first equation.

## Advantages
- **Direct Solution**: Solves the system in a finite number of steps without iteration.
- **Standard Procedure**: Works well for small to medium-sized systems.

## Limitations
- **Numerical Instability**: Can be sensitive to round-off errors, especially for large systems.
- **Pivoting Needed**: Partial or full pivoting is often necessary to improve stability and avoid division by zero.

## Usage

To use the Gauss Elimination Method:
1. Set up the augmented matrix \( [A|b] \).
2. Perform forward elimination to transform \( A \) into an upper triangular matrix.
3. Use back substitution to solve for each variable in \( x \).


# Gauss-Jordan Elimination

The Gauss-Jordan Elimination Method is a direct, non-iterative algorithm for solving systems of linear equations. This method extends Gauss Elimination by transforming the matrix into **reduced row echelon form** (RREF), allowing for a more straightforward solution without requiring back substitution.

## Problem Setup

We aim to solve a system of linear equations represented by:

\[
Ax = b
\]

where:
- \( A \) is a square matrix of coefficients,
- \( x \) is the vector of unknowns,
- \( b \) is the result vector.

## Steps of the Gauss-Jordan Elimination Method

The Gauss-Jordan method consists of **Forward Elimination** and **Backward Elimination** phases to convert the matrix into reduced row echelon form.

### 1. Forward Elimination

1. **Pivot Selection**  
   For each row \( i \), select a pivot element \( a_{ii} \) (typically the largest absolute value in the column) to reduce computational error.

2. **Row Scaling**  
   Divide each row \( i \) by the pivot element \( a_{ii} \) to make it equal to 1:

   \[
   \text{Row } i = \frac{\text{Row } i}{a_{ii}}
   \]

3. **Row Operations to Zero Out Entries Below the Pivot**  
   For each subsequent row \( j > i \), eliminate the entry \( a_{ji} \) by using the row operation:

   \[
   \text{Row } j = \text{Row } j - a_{ji} \times \text{Row } i
   \]

   This process zeros out all entries below the pivot in each column.

### 2. Backward Elimination

Once the matrix is in an upper triangular form, the goal is to make the matrix diagonal with each diagonal entry equal to 1.

1. **Row Operations to Zero Out Entries Above Each Pivot**  
   For each row \( i \), use row operations to make all entries above the pivot \( a_{ii} \) equal to 0:

   \[
   \text{Row } k = \text{Row } k - a_{ki} \times \text{Row } i \quad \text{for } k < i
   \]

2. Repeat this process until the entire matrix \( A \) is transformed into the **identity matrix**.

### 3. Solution Extraction

At the end of the process, the augmented matrix is in the form:

\[
[I | x]
\]

where \( I \) is the identity matrix, and \( x \) is the solution vector.

### Step-by-Step Solution

1. **Form the Augmented Matrix**  
   Set up the augmented matrix:

   \[
   \begin{bmatrix}
   1 & 1 & 1 & | & 6 \\
   0 & 2 & 5 & | & -4 \\
   2 & 5 & -1 & | & 27 \\
   \end{bmatrix}
   \]

2. **Forward and Backward Elimination**  
   - Use row operations to create zeros below and above each pivot, turning the augmented matrix into the form \( [I | x] \).

3. **Solution Extraction**  
   The rightmost column gives the solution vector \( x \).

## Advantages
- **Direct Solution in RREF**: Eliminates the need for back substitution.
- **Systematic**: Straightforward to apply and always yields the solution (if it exists).

## Limitations
- **Numerical Instability**: Can be sensitive to round-off errors.
- **More Operations Required**: May involve more row operations compared to the Gauss Elimination method, especially for large systems.

## Usage

To use the Gauss-Jordan Elimination Method:
1. Set up the augmented matrix \( [A|b] \).
2. Perform forward elimination and backward elimination to transform \( A \) into the identity matrix.
3. Read off the solution vector \( x \) from the final column.


# LU factorization

LU Factorization (or LU Decomposition) is a method for solving systems of linear equations by decomposing a given matrix \( A \) into the product of a **lower triangular matrix** \( L \) and an **upper triangular matrix** \( U \). This decomposition simplifies the process of solving multiple systems with the same coefficient matrix, especially for large systems.

## Problem Setup

For a square matrix \( A \), we aim to find matrices \( L \) and \( U \) such that:

\[
A = LU
\]

where:
- \( L \) is a lower triangular matrix (all entries above the main diagonal are zero, and diagonal entries are often set to 1),
- \( U \) is an upper triangular matrix (all entries below the main diagonal are zero).

Given \( A = LU \), we can solve \( Ax = b \) in two steps:
1. Solve \( Ly = b \) for \( y \) (using **forward substitution**).
2. Solve \( Ux = y \) for \( x \) (using **back substitution**).

## Steps of the LU Factorization Method

1. **Decompose Matrix \( A \) into \( L \) and \( U \)**
   - Begin with \( A \) and perform Gaussian elimination to create zeros below the main diagonal.
   - The factors used to eliminate entries in \( A \) form the elements of \( L \), while the resulting upper triangular form becomes \( U \).

2. **Solve the System \( Ly = b \) Using Forward Substitution**
   - Since \( L \) is lower triangular, solve for each \( y_i \) in order, using only the values from previous rows.

3. **Solve the System \( Ux = y \) Using Back Substitution**
   - Since \( U \) is upper triangular, solve for each \( x_i \) starting from the last row and moving upward.

### Step-by-Step Solution

1. **LU Decomposition**:
   - Decompose \( A \) into \( L \) and \( U \) such that:

   \[
   L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 1 & 3 & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 2 & 3 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}
   \]

2. **Forward Substitution**:
   - Solve \( Ly = b \).

3. **Back Substitution**:
   - Solve \( Ux = y \).

## Advantages
- **Efficient for Repeated Solutions**: Useful when solving multiple systems with the same matrix \( A \) but different \( b \) vectors.
- **Stable for Certain Matrices**: Provides a stable factorization when \( A \) is well-conditioned.

## Limitations
- **Pivoting May Be Required**: Partial pivoting may be necessary to ensure stability for some matrices.
- **Square Matrices Only**: LU decomposition is defined primarily for square matrices, though it can be extended to rectangular ones under certain conditions.

## Usage

To use the LU Factorization Method:
1. Factorize \( A \) into \( L \) and \( U \).
2. Use forward substitution to solve \( Ly = b \).
3. Use back substitution to solve \( Ux = y \).


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


# Differential Equations

# Runge-Kutta Method

The Runge-Kutta Method is a numerical technique for solving ordinary differential equations (ODEs). Among various forms of this method, the **Fourth-Order Runge-Kutta Method (RK4)** is widely used due to its balance between accuracy and computational efficiency.

## Problem Setup

We aim to solve an initial value problem of the form:

\[
\frac{dy}{dx} = f(x, y), \quad y(x_0) = y_0
\]

The goal is to approximate \( y(x) \) over a specified interval, given the initial condition \( y(x_0) = y_0 \).

## Fourth-Order Runge-Kutta Method (RK4) Steps

The RK4 method uses four estimates of the slope within each interval to approximate \( y \) at each step. For a given step size \( h \), starting from \( x_n \) and \( y_n \):

1. **Calculate Intermediate Slopes**:
   - \( k_1 = h f(x_n, y_n) \)
   - \( k_2 = h f\left(x_n + \frac{h}{2}, y_n + \frac{k_1}{2}\right) \)
   - \( k_3 = h f\left(x_n + \frac{h}{2}, y_n + \frac{k_2}{2}\right) \)
   - \( k_4 = h f(x_n + h, y_n + k_3) \)

2. **Update the Solution**:
   - Calculate the next value \( y_{n+1} \) using:

     \[
     y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
     \]

3. **Advance to the Next Point**:
   - Increment \( x \) by \( h \): \( x_{n+1} = x_n + h \)

4. **Repeat the Steps**:
   - Continue this process over the desired interval to approximate the solution \( y(x) \) at each step.

## Advantages
- **High Accuracy**: RK4 provides a good balance of accuracy without requiring excessively small step sizes.
- **Stability**: More stable than lower-order methods for many types of ODEs.

## Limitations
- **Computational Intensity**: Requires multiple function evaluations per step, making it slower than simpler methods for large systems.
- **Step Size Dependency**: Accuracy depends on the step size \( h \); smaller \( h \) increases accuracy but also computation time.

## Usage

To use the Runge-Kutta Method:
1. Define the function \( f(x, y) \) and initial conditions.
2. Choose a step size \( h \) and an interval over which to compute the solution.
3. Apply the RK4 steps iteratively to approximate \( y \) at each point.

# Matrix Inversion Method

Matrix inversion is the process of finding the **inverse** of a square matrix \( A \), denoted \( A^{-1} \), such that:

\[
A \cdot A^{-1} = I
\]

where \( I \) is the identity matrix. If a matrix has an inverse, it is called **invertible** or **non-singular**. Matrix inversion is widely used to solve systems of linear equations and in various applications in mathematics, physics, and engineering.

## Conditions for Invertibility

A square matrix \( A \) is invertible if:
1. \( A \) is a **square matrix** (same number of rows and columns).
2. The **determinant** of \( A \) is non-zero, i.e., \( \det(A) \neq 0 \).

If \( A \) is not invertible, it is called a **singular matrix** and does not have an inverse.

## Methods for Matrix Inversion

### 1. Gauss-Jordan Elimination

The most common way to invert a matrix is by using the Gauss-Jordan elimination method. This involves transforming \( A \) into the identity matrix while applying the same row operations to an identity matrix of the same size, ultimately producing \( A^{-1} \).

#### Steps:
1. **Set Up the Augmented Matrix**  
   Start with an augmented matrix \([A | I]\), where \( I \) is the identity matrix of the same size as \( A \).

2. **Row Operations**  
   Apply row operations to transform \( A \) into the identity matrix \( I \). Perform the same operations on \( I \) as well.

3. **Result**  
   Once \( A \) has been transformed into \( I \), the augmented portion will become \( A^{-1} \).

Example:

For a matrix \( A = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix} \),

- Set up the augmented matrix:  
  \[
  \left[ \begin{array}{cc|cc} 2 & 1 & 1 & 0 \\ 5 & 3 & 0 & 1 \end{array} \right]
  \]

- Apply row operations until the left side is the identity matrix, and the right side will yield \( A^{-1} \).

### 2. Adjoint Method (for 2x2 or 3x3 Matrices)

For small matrices, \( A^{-1} \) can also be calculated using the **adjoint formula**:

\[
A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
\]

where \( \text{adj}(A) \) is the adjugate of \( A \), and \( \det(A) \) is the determinant. This method is practical for 2x2 or 3x3 matrices but becomes computationally expensive for larger matrices.

### 3. LU Decomposition (for Large Matrices)

For larger matrices, **LU decomposition** can be used to find \( A^{-1} \) by decomposing \( A \) into a product of a lower triangular matrix \( L \) and an upper triangular matrix \( U \). This approach is more efficient for computational purposes.

## Example for 2x2 Matrix

For a 2x2 matrix:

\[
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\]

The inverse \( A^{-1} \) (if \( \det(A) = ad - bc \neq 0 \)) is given by:

\[
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\]

## Applications

Matrix inversion is used in:
- Solving linear systems of equations \( Ax = b \), where \( x = A^{-1} b \).
- Linear transformations and coordinate transformations.
- Engineering and physics problems involving control systems, dynamics, and optimization.

## Advantages and Limitations

### Advantages
- Provides an exact solution when the matrix is invertible.
- Useful in solving systems of linear equations and various applications in linear algebra.

### Limitations
- Computationally intensive for large matrices.
- Not all matrices have inverses; only non-singular matrices can be inverted.
- Numerically unstable for matrices with very small determinants.



