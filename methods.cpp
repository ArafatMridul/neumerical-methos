#include <bits/stdc++.h>
using namespace std;

// Solution of Linear Equations Functions

void jacobiIterativeMethod()
{
    cout << "Jacobi iterative method code here" << endl;
}

void gaussSeidelIterativeMethod()
{
    cout << "Gauss-Seidel iterative method code here" << endl;
}

void gaussEliminationMethod()
{
    cout << "Gauss Elimination method code here" << endl;
}

void gaussJordanEliminationMethod()
{
    cout << "Gauss-Jordan Elimination method code here" << endl;
}

void LUFactorizationMethod()
{
    cout << "LU factorization method code here" << endl;
}

// Solution of Non-linear Equations functions
double evaluateBisectionPolynomial(double x, const std::vector<double> &coefficients)
{
    double result = 0.0;
    for (int i = 0; i < coefficients.size(); ++i)
    {
        result += coefficients[i] * std::pow(x, i);
    }
    return result;
}
double bisectionMainMethod(const std::vector<double> &coefficients, double a, double b, double tolerance)
{
    double fa = evaluateBisectionPolynomial(a, coefficients);
    double fb = evaluateBisectionPolynomial(b, coefficients);
    if (fa * fb >= 0)
    {
        std::cerr << "fa and fb must have diff signs.Bisection Method Fail" << std::endl;
        return NAN;
    }
    double c;
    while ((b - a) >= tolerance)
    {
        c = (a + b) / 2;
        double fc = evaluateBisectionPolynomial(c, coefficients);
        if (fc == 0.0)
        {
            break;
        }
        if (fc * fa < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }
    }
    return c;
}
std::vector<double> FindTheRoots(const std::vector<double> &coefficients, double start, double end, double step, double tolerance)
{
    std::vector<double> roots;
    double prev = evaluateBisectionPolynomial(start, coefficients);
    for (double x = start; x <= end; x += step)
    {
        double current = evaluateBisectionPolynomial(x, coefficients);
        if (prev * current < 0)
        {
            double root = bisectionMainMethod(coefficients, x - step, x, tolerance);
            roots.push_back(root);
        }
        prev = current;
    }
    return roots;
}

void biSectionMethod()
{
    cout << "Bi-section method code here" << endl;
    int degree;
    std::cout << "Enter the degree of the polynomial: ";
    std::cin >> degree;

    std::vector<double> coefficients(degree + 1);
    std::cout << "Enter the coefficient Lowest Degree to Highest Degree:" << std::endl;
    for (int i = 0; i <= degree; ++i)
    {
        std::cin >> coefficients[i];
    }
    double start = -10;
    double end = 10;
    double step = 0.1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    std::vector<double> roots = FindTheRoots(coefficients, start, end, step, tolerance);

    std::cout << "Roots found:" << std::endl;
    for (double root : roots)
    {
        std::cout << root << std::endl;
    }
}
double falsePositionMainMethod(const std::vector<double> &coefficients, double a, double b, double tolerance)
{
    double fa = evaluateBisectionPolynomial(a, coefficients);
    double fb = evaluateBisectionPolynomial(b, coefficients);
    if (fa * fb >= 0)
    {
        std::cerr << "It should have different signs.False Position method Failure" << std::endl;
        return  NAN;
    }
    double c;
    while (std::fabs(b - a) >= tolerance)
    {
        c = (a * fb - b * fa) / (fb - fa);
        double fc = evaluateBisectionPolynomial(c, coefficients);

        if (fc == 0.0)
        {
            break;
        }
        if (fc * fa < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }
    }
    return c;
}
std::vector<double> FindTheFalsePositionRoots(const std::vector<double> &coefficients, double start, double end, double step, double tol)
{
    std::vector<double> roots;
    double prev = evaluateBisectionPolynomial(start, coefficients);

    for (double x = start; x <= end; x += step)
    {
        double current = evaluateBisectionPolynomial(x, coefficients);

        if (prev * current < 0)
        {

            double root = falsePositionMainMethod(coefficients, x - step, x, tol);
            roots.push_back(root);
        }

        prev = current;
    }

    return roots;
}

void falsePositionMethod()
{
    cout << "False position method code here" << endl;
    int degree;
    std::cout << "Enter the degree of the polynomial: ";
    std::cin >> degree;
    std::vector<double> coefficients(degree + 1);
    std::cout << "Enter the coefficients Lowest to Highest Degree Consecutively:" << std::endl;
    for (int i = 0; i <= degree; ++i)
    {
        std::cin >> coefficients[i];
    }
    double start = -10;
    double end = 10;
    double step = 0.1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    std::vector<double> roots = FindTheFalsePositionRoots(coefficients, start, end, step, tolerance);
    std::cout << "Roots found:" << std::endl;
    for (double root : roots)
    {
        std::cout << root << std::endl;
    }
}
double evaluateEquation(const std::string &equationStr, double x)
{
    double result = 0.0;
    std::istringstream iss(equationStr);
    char op = '+';
    while (iss)
    {
        double coeff = 1.0;
        int power = 0;
        char c = iss.peek();
        if (c == 'x')
        {
            iss.get();
            if (iss.peek() == '^')
            {
                iss.get();
                iss >> power;
            }
            else
            {
                power = 1;
            }
        }
        else if (isdigit(c) || c == '.')
        {
            iss >> coeff;
            if (iss.peek() == 'x')
            {
                iss.get();
                if (iss.peek() == '^')
                {
                    iss.get();
                    iss >> power;
                }
                else
                {
                    power = 1;
                }
            }
        }
        else
        {
            iss.get();
            continue;
        }

        double term = coeff * pow(x, power);
        if (op == '+')
        {
            result += term;
        }
        else if (op == '-')
        {
            result -= term;
        }

        iss >> op;
    }
    return result;
}

double secantMainMethod(const std::string &equationStr, double x0, double x1, double tolerance, int maxIterations)
{
    double f0 = evaluateEquation(equationStr, x0);
    double f1 = evaluateEquation(equationStr, x1);

    for (int i = 0; i < maxIterations; ++i)
    {
        if (std::abs(f1 - f0) < 1e-10)
        {
            std::cerr << "Function values are too close,and so stopping iteration." << std::endl;
            return NAN;
        }

        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        double f2 = evaluateEquation(equationStr, x2);

        if (std::abs(x2 - x1) < tolerance)
        {
            return x2;
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }

    std::cerr << "Maximum iterations reached without converging ." << std::endl;
    return x1;
}

void secantMethod()
{
    cout << "Secant method code here" << endl;
    std::string equation;
    double x0, x1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    int maxIterations;
    cout << "Enter the maximum number of iterations: ";
    cin >> maxIterations;

    std::cout << "Enter the polynomial equation (Enter the equation in string format): ";
    std::cin.ignore();
    std::getline(std::cin, equation);

    std::cout << "Enter the first initial guess: ";
    std::cin >> x0;

    std::cout << "Enter the second initial guess: ";
    std::cin >> x1;

    double root = secantMainMethod(equation, x0, x1, tolerance, maxIterations);
    if (!std::isnan(root))
    {
        std::cout << "The root is: " << root << std::endl;
    }
    else
    {
        std::cout << "Failed in finding a root." << std::endl;
    }
}
double evaluateNewtonDerivative(const std::string &equationStr, double x)
{
    double result = 0.0;
    std::istringstream iss(equationStr);
    char op = '+';
    while (iss)
    {
        double coeff = 1.0;
        int power = 0;
        char c = iss.peek();
        if (c == 'x')
        {
            iss.get();
            if (iss.peek() == '^')
            {
                iss.get();
                iss >> power;
            }
            else
            {
                power = 1;
            }
        }
        else if (isdigit(c) || c == '.')
        {
            iss >> coeff;
            if (iss.peek() == 'x')
            {
                iss.get();
                if (iss.peek() == '^')
                {
                    iss.get();
                    iss >> power;
                }
                else
                {
                    power = 1;
                }
            }
        }
        else
        {
            iss.get();
            continue;
        }

        if (power > 0)
        {
            double term = coeff * power * pow(x, power - 1);
            if (op == '+')
            {
                result += term;
            }
            else if (op == '-')
            {
                result -= term;
            }
        }

        iss >> op;
    }
    return result;
}

double newtonRaphsonMainMethod(const std::string &equationStr, double initialGuess, double tolerance, int maxIterations)
{
    double x = initialGuess;
    for (int i = 0; i < maxIterations; ++i)
    {
        double fx = evaluateEquation(equationStr, x);
        double dfx = evaluateNewtonDerivative(equationStr, x);

        if (std::abs(dfx) < 1e-10)
        {
            std::cerr << "Derivative is too small,so stopping iteration." << std::endl;
            return NAN;
        }

        double x_new = x - fx / dfx;

        if (std::abs(x_new - x) < tolerance)
        {
            return x_new;
        }

        x = x_new;
    }

    std::cerr << "Maximum iterations reached without converging ." << std::endl;
    return x;
}

void newtonRaphsonMethod()
{
    cout << "Newton-Raphson method code here" << endl;
    std::string equation;
    double initialGuess;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;

    int maxIterations;
    cout << "Enter the maximum number of iterations: ";
    cin >> maxIterations;
    std::cout << "Enter the polynomial equation (Enter the equation in string format): ";
    std::cin.ignore();
    std::getline(std::cin, equation);
    std::cout << "Enter the initial guess: ";
    std::cin >> initialGuess;
    double root = newtonRaphsonMainMethod(equation, initialGuess, tolerance, maxIterations);
    if (!std::isnan(root))
    {
        std::cout << "The root is: " << root << std::endl;
    }
    else
    {
        std::cout << "Failed in finding a root." << std::endl;
    }
}

// Solution of Differential Equations Funtions

double evaluateDerivative(vector<double> &coefficients, double x)
{
    double result = 0;
    int degree = coefficients.size() - 1;

    for (int i = 0; i < degree; ++i)
    {
        result += coefficients[i] * (degree - i) * pow(x, degree - i - 1);
    }
    return result;
}

void rungeKuttaMethod()
{
    int degree;
    cout << "Enter the degree of the polynomial: ";
    cin >> degree;

    // Read coefficients from user
    vector<double> coefficients(degree + 1);
    cout << "Enter the coefficients (from highest degree to lowest):\n";
    for (int i = 0; i <= degree; ++i)
    {
        cin >> coefficients[i];
    }

    double x0, y0, x, h;
    cout << "Enter initial values of x0 and y0: ";
    cin >> x0 >> y0;
    cout << "Enter the x value to find y at: ";
    cin >> x;
    cout << "Enter step size h: ";
    cin >> h;

    int n = (int)((x - x0) / h); // Number of steps
    double y = y0;

    for (int i = 0; i < n; i++)
    {
        double k1 = h * evaluateDerivative(coefficients, x0);
        double k2 = h * evaluateDerivative(coefficients, x0 + h / 2);
        double k3 = h * evaluateDerivative(coefficients, x0 + h / 2);
        double k4 = h * evaluateDerivative(coefficients, x0 + h);

        // Update next value of y
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;

        // Move to next point
        x0 += h;
    }
    cout << "The value of y at x = " << x << " is: " << y << "\n";
}

// Matrix Inversion Functions

void matrixInversion()
{
    int n;
    cout << "Enter the order of the matrix: ";
    cin >> n;

    vector<vector<double>> a(n, vector<double>(n));

    cout << "Enter the matrix elements:" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> a[i][j];
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n, 0));

    // Initialize inverse as an identity matrix
    for (int i = 0; i < n; i++)
    {
        inverse[i][i] = 1;
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++)
    {
        double diag = a[i][i];
        for (int j = 0; j < n; j++)
        {
            a[i][j] /= diag;
            inverse[i][j] /= diag;
        }

        for (int k = 0; k < n; k++)
        {
            if (k != i)
            {
                double factor = a[k][i];
                for (int j = 0; j < n; j++)
                {
                    a[k][j] -= factor * a[i][j];
                    inverse[k][j] -= factor * inverse[i][j];
                }
            }
        }
    }

    // Display the inverse matrix
    cout << "Inverse matrix:" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << inverse[i][j] << " ";
        }
        cout << endl;
    }
}

void menu()
{
    int type_choice;
    cout << "----- Neumerical Methods -----" << endl;
    cout << "1. Solution of Linear Equations." << endl;
    cout << "2. Solution of Non-linear Equations." << endl;
    cout << "3. Solution of Differential Equations." << endl;
    cout << "4. Matrix Inversion." << endl;
    cout << "5. exit." << endl;

    cout << "Enter your choice : ";
    cin >> type_choice;
    char method_choice;

    switch (type_choice)
    {
    case 1:
        cout << "a. Jacobi iterative method." << endl;
        cout << "b. Gauss-Seidel iterative method." << endl;
        cout << "c. Gauss Elimination method." << endl;
        cout << "d. Gauss-Jordan Elimination method." << endl;
        cout << "e. LU factorization method." << endl;

        cout << "Enter your choice: ";
        cin >> method_choice;

        switch (method_choice)
        {
        case 'a':
            jacobiIterativeMethod();
            menu();
            break;
        case 'b':
            gaussSeidelIterativeMethod();
            menu();
            break;
        case 'c':
            gaussEliminationMethod();
            menu();
            break;
        case 'd':
            gaussJordanEliminationMethod();
            menu();
            break;
        case 'e':
            LUFactorizationMethod();
            menu();
            break;
        default:
            cout << "Choose correct option." << endl;
            menu();
        }
        break;

    case 2:
        cout << "a. Bi-section method." << endl;
        cout << "b. False position method." << endl;
        cout << "c. Secant method." << endl;
        cout << "d. Newton-Raphson method." << endl;

        cout << "Enter your choice: ";
        cin >> method_choice;

        switch (method_choice)
        {
        case 'a':
            biSectionMethod();
            menu();
            break;
        case 'b':
            falsePositionMethod();
            menu();
            break;
        case 'c':
            secantMethod();
            menu();
            break;
        case 'd':
            newtonRaphsonMethod();
            menu();
            break;
        default:
            cout << "Choose correct option." << endl;
            menu();
        }
        break;

    case 3:
        cout << "a. Runge-Kutta Method" << endl;

        cout << "Enter your choice: ";
        cin >> method_choice;

        switch (method_choice)
        {
        case 'a':
            rungeKuttaMethod();
            menu();
            break;
        }
        break;

    case 4:
        cout << "Matrix Inversion : " << endl;
        matrixInversion();
        menu();
        break;
    case 5:
        return;
        break;

    default:
        cout << "Choose correct option." << endl;
        menu();
    }
}

int main()
{

    menu();

    return 0;
}