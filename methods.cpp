#include <bits/stdc++.h>
using namespace std;

// Solution of Linear Equations Functions

void jacobiIterativeMethod() {
    cout<<"enter the number of variable:\n";
    int n;
    cin>>n;
    cout<<"Next "<<n<<"lines takes the "<<n+1<<"coefficinets of the variables sequencially.\n";
    cout<<"For a equation a11x1 + a22x2 = b1  input format is a11 a11 b1\n";
    vector<vector<long double>>matrix;
    matrix.assign(n,vector<long double>(n+1));
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<=n; j++)
        {
            cin>>matrix[i][j];
        }
    }
    long double err=1e-5;
    vector<long double>cur(n),nxt(n);
    for(int i=0; i<n; i++)cur[i]=0;
    for(int i=0; i<n; i++)
    {
        if(matrix[i][i]==0)
        {
            for(int j=i+1; j<n; j++)
            {
                if(matrix[j][i]!=0)
                {
                    swap(matrix[i],matrix[j]);
                }
            }
        }
    }
    long double dif=100;
    while(dif>err)
    {
        for(int i=0; i<n; i++)
        {
            nxt[i]=matrix[i][n];
            for(int j=0; j<n; j++)
            {
                if(i!=j)nxt[i]=nxt[i]-matrix[i][j]*cur[j];
            }
            nxt[i]=nxt[i]/matrix[i][i];
        }
        dif=0;
        for(int i=0;i<n;i++)
        {
            dif=max(dif,abs(cur[i]-nxt[i]));
            cur[i]=nxt[i];
        }
    }
    for(int i=0;i<n;i++)cout<<cur[i]<<" "; cout<<"\n";
}
// Gauss Seidel elimination
void gaussSeidelIterativeMethod() {
    long double err=1e-5;
    cout<<"enter the number of variable:\n";
    int n;
    cin>>n;
    cout<<"Next "<<n<<"lines takes the "<<n+1<<"coefficinets of the variables sequencially.\n";
    cout<<"For a equation a11x1 + a22x2 = b1  input format is a11 a11 b1\n";
    vector<vector<long double>>matrix;
    matrix.assign(n,vector<long double>(n+1));
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<=n; j++)
        {
            cin>>matrix[i][j];
        }
    }
    vector<long double>cur(n);
    for(int i=0; i<n; i++)cur[i]=0;
    for(int i=0; i<n; i++)
    {
        if(matrix[i][i]==0)
        {
            for(int j=i+1; j<n; j++)
            {
                if(matrix[j][i]!=0)
                {
                    swap(matrix[i],matrix[j]);
                }
            }
        }
    }
    long double dif=100;
    while(dif>err)
    {
        dif=0;
        for(int i=0; i<n; i++)
        {
            long double pre=cur[i];
            cur[i]=matrix[i][n];
            for(int j=0; j<n; j++)
            {
                if(i!=j)cur[i]=cur[i]-matrix[i][j]*cur[j];
            }
            cur[i]=cur[i]/matrix[i][i];
            dif=max(dif,abs(cur[i]-pre));
        }
    }
    for(int i=0;i<n;i++)cout<<cur[i]<<" "; cout<<"\n";
}
// Gauss Elimination Method

void format(int n,vector<vector<long double>>&matrix)
{
    long double err=1e-5;
    for(int i=0; i<n; i++)
    {
        if(abs(matrix[i][i])<=0)
        {
            for(int j=i+1; j<n; j++)
            {
                if(abs(matrix[j][i])>err)
                {
                    swap(matrix[i],matrix[j]);
                }
            }
        }
    }
}
void gaussEliminationMethod() {
    long double err=1e-5;
    cout<<"enter the number of variable:\n";
    int n;
    cin>>n;
    cout<<"Next "<<n<<"lines takes the "<<n+1<<"coefficinets of the variables sequencially.\n";
    cout<<"For a equation a11x1 + a22x2 = b1  input format is a11 a11 b1\n";
    vector<vector<long double>>matrix;
    matrix.assign(n,vector<long double>(n+1));
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<=n; j++)
        {
            cin>>matrix[i][j];
        }
    }
    for(int j=0; j<n-1; j++)
    {
        format(n,matrix);
        for(int i=j+1; i<n; i++)
        {
            if(abs(matrix[i][j])>=err)
            {
                long double mag=-matrix[i][j]/matrix[j][j];
                for(int k=j; k<=n; k++)
                {
                    matrix[i][k]=matrix[i][k]+mag*matrix[j][k];
                }
            }
        }
    }
    vector<long double>ans(n);
    for(int i=n-1;i>=0;i--)
    {
        ans[i]=matrix[i][n];
        for(int j=i+1;j<n;j++)
        {
            ans[i]=ans[i]-matrix[i][j]*ans[j];
        }
        ans[i]=ans[i]/matrix[i][i];
    }
    for(auto it:ans)cout<<it<<" "; cout<<"\n";
}

void gaussJordanEliminationMethod()
{
        long double err=1e-5;
    cout<<"enter the number of variable:\n";
    int n;
    cin>>n;
    cout<<"Next "<<n<<"lines takes the "<<n+1<<"coefficinets of the variables sequencially.\n";
    cout<<"For a equation a11x1 + a22x2 = b1  input format is a11 a11 b1\n";
    vector<vector<long double>>matrix;
    matrix.assign(n,vector<long double>(n+1));
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<=n; j++)
        {
            cin>>matrix[i][j];
        }
    }
    for(int j=0; j<n-1; j++)
    {
        format(n,matrix);
        for(int i=j+1; i<n; i++)
        {
            if(abs(matrix[i][j])>=err)
            {
                long double mag=-matrix[i][j]/matrix[j][j];
                for(int k=j; k<=n; k++)
                {
                    matrix[i][k]=matrix[i][k]+mag*matrix[j][k];
                }
            }
        }
    }
    for(int j=n-1; j>0; j--)
    {
        for(int i=j-1; i>=0; i--)
        {
            if(abs(matrix[i][j])>=err)
            {
                long double mag=-matrix[i][j]/matrix[j][j];
                for(int k=j; k>i; k--)
                {
                    matrix[i][k]=matrix[i][k]+mag*matrix[j][k];
                }
                matrix[i][n]=matrix[i][n]+mag*matrix[j][n];
            }
        }
    }
    vector<long double>ans(n);
    for(int i=0;i<n;i++)ans[i]=matrix[i][n]/matrix[i][i];
    for(auto it:ans)cout<<it<<" "; cout<<"\n";
}

// LU Factorization

void luFactorization(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    int n = A.size();
    
    // Initialize L and U matrices
    L.resize(n, vector<double>(n, 0));
    U.resize(n, vector<double>(n, 0));
    
    for (int i = 0; i < n; i++) {
        // Upper Triangular U
        for (int j = i; j < n; j++) {
            U[i][j] = A[i][j];
            for (int k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
        
        // Lower Triangular L
        for (int j = i; j < n; j++) {
            if (i == j) {
                L[i][i] = 1; // Diagonal elements are set to 1
            } else {
                L[j][i] = A[j][i];
                for (int k = 0; k < i; k++) {
                    L[j][i] -= L[j][k] * U[k][i];
                }
                L[j][i] /= U[i][i];
            }
        }
    }
}

// Function to solve Ly = b using forward substitution
vector<double> forwardSubstitution(const vector<vector<double>>& L, const vector<double>& b) {
    int n = L.size();
    vector<double> y(n);

    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
    return y;
}

// Function to solve Ux = y using backward substitution
vector<double> backwardSubstitution(const vector<vector<double>>& U, const vector<double>& y) {
    int n = U.size();
    vector<double> x(n);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }
    return x;
}

// Function to print a matrix
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double value : row) {
            cout << setw(10) << value << " ";
        }
        cout << endl;
    }
}

void LUFactorizationMethod()
{
    int n;

    cout << "Enter the size of the matrix (n x n): ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    cout << "Enter the elements of the matrix A: " << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    cout << "Enter the elements of the vector b:" << endl;
    for (int i = 0; i < n; i++) {
        cout << "b[" << i << "] = ";
        cin >> b[i];
    }

    vector<vector<double>> L, U;

    // Perform LU Factorization
    luFactorization(A, L, U);

    cout << "Matrix L:" << endl;
    printMatrix(L);

    cout << "Matrix U:" << endl;
    printMatrix(U);

    // Solve Ly = b
    vector<double> y = forwardSubstitution(L, b);

    cout << "Intermediate vector y after forward substitution:" << endl;
    for (double val : y) {
        cout << val << " ";
    }
    cout << endl;

    // Solve Ux = y
    vector<double> x = backwardSubstitution(U, y);

    cout << "Solution vector x:" << endl;
    for (double val : x) {
        cout << val << " ";
    }
    cout << endl;
}

// Solution of Non-linear Equations functions
double evaluateBisectionPolynomial(double x, vector<double> &coefficients)
{
    double result = 0.0;
    for (int i = 0; i < coefficients.size(); ++i)
    {
        result += coefficients[i] * pow(x, i);
    }
    return result;
}
double bisectionMainMethod(vector<double> &coefficients, double a, double b, double tolerance)
{
    double fa = evaluateBisectionPolynomial(a, coefficients);
    double fb = evaluateBisectionPolynomial(b, coefficients);
    if (fa * fb >= 0)
    {
        cerr << "fa and fb must have diff signs.Bisection Method Fail" << endl;
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
vector<double> FindTheRoots(vector<double> &coefficients, double start, double end, double step, double tolerance)
{
    vector<double> roots;
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
    cout << "Enter the degree of the polynomial: ";
    cin >> degree;

    vector<double> coefficients(degree + 1);
    cout << "Enter the coefficient Lowest Degree to Highest Degree:" << endl;
    for (int i = 0; i <= degree; ++i)
    {
        cin >> coefficients[i];
    }
    double start = -10;
    double end = 10;
    double step = 0.1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    vector<double> roots = FindTheRoots(coefficients, start, end, step, tolerance);

    cout << "Roots found:" << endl;
    for (double root : roots)
    {
        cout << root << endl;
    }
}
double falsePositionMainMethod(vector<double> &coefficients, double a, double b, double tolerance)
{
    double fa = evaluateBisectionPolynomial(a, coefficients);
    double fb = evaluateBisectionPolynomial(b, coefficients);
    if (fa * fb >= 0)
    {
        cerr << "It should have different signs.False Position method Failure" << endl;
        return  NAN;
    }
    double c;
    while (fabs(b - a) >= tolerance)
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
vector<double> FindTheFalsePositionRoots(vector<double> &coefficients, double start, double end, double step, double tol)
{
    vector<double> roots;
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
    cout << "Enter the degree of the polynomial: ";
    cin >> degree;
    vector<double> coefficients(degree + 1);
    cout << "Enter the coefficients Lowest to Highest Degree Consecutively:" << endl;
    for (int i = 0; i <= degree; ++i)
    {
        cin >> coefficients[i];
    }
    double start = -10;
    double end = 10;
    double step = 0.1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    vector<double> roots = FindTheFalsePositionRoots(coefficients, start, end, step, tolerance);
    cout << "Roots found:" << endl;
    for (double root : roots)
    {
        cout << root << endl;
    }
}
double evaluateEquation(string &equationStr, double x)
{
    double result = 0.0;
    istringstream iss(equationStr);
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

double secantMainMethod(string &equationStr, double x0, double x1, double tolerance, int maxIterations)
{
    double f0 = evaluateEquation(equationStr, x0);
    double f1 = evaluateEquation(equationStr, x1);

    for (int i = 0; i < maxIterations; ++i)
    {
        if (abs(f1 - f0) < 1e-10)
        {
            cerr << "Function values are too close,and so stopping iteration." << endl;
            return NAN;
        }

        double x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        double f2 = evaluateEquation(equationStr, x2);

        if (abs(x2 - x1) < tolerance)
        {
            return x2;
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }

    cerr << "Maximum iterations reached without converging ." << endl;
    return x1;
}

void secantMethod()
{
    cout << "Secant method code here" << endl;
    string equation;
    double x0, x1;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;
    int maxIterations;
    cout << "Enter the maximum number of iterations: ";
    cin >> maxIterations;

    cout << "Enter the polynomial equation (Enter the equation in string format): ";
    cin.ignore();
    getline(cin, equation);

    cout << "Enter the first initial guess: ";
    cin >> x0;

    cout << "Enter the second initial guess: ";
    cin >> x1;

    double root = secantMainMethod(equation, x0, x1, tolerance, maxIterations);
    if (!isnan(root))
    {
        cout << "The root is: " << root << endl;
    }
    else
    {
        cout << "Failed in finding a root." << endl;
    }
}
double evaluateNewtonDerivative(string &equationStr, double x)
{
    double result = 0.0;
    istringstream iss(equationStr);
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

double newtonRaphsonMainMethod(string &equationStr, double initialGuess, double tolerance, int maxIterations)
{
    double x = initialGuess;
    for (int i = 0; i < maxIterations; ++i)
    {
        double fx = evaluateEquation(equationStr, x);
        double dfx = evaluateNewtonDerivative(equationStr, x);

        if (abs(dfx) < 1e-10)
        {
            cerr << "Derivative is too small,so stopping iteration." << endl;
            return NAN;
        }

        double x_new = x - fx / dfx;

        if (abs(x_new - x) < tolerance)
        {
            return x_new;
        }

        x = x_new;
    }

    cerr << "Maximum iterations reached without converging ." << endl;
    return x;
}

void newtonRaphsonMethod()
{
    cout << "Newton-Raphson method code here" << endl;
    string equation;
    double initialGuess;
    double tolerance;
    cout << "Enter the tolerance: ";
    cin >> tolerance;

    int maxIterations;
    cout << "Enter the maximum number of iterations: ";
    cin >> maxIterations;
    cout << "Enter the polynomial equation (Enter the equation in string format): ";
    cin.ignore();
    getline(cin, equation);
    cout << "Enter the initial guess: ";
    cin >> initialGuess;
    double root = newtonRaphsonMainMethod(equation, initialGuess, tolerance, maxIterations);
    if (!isnan(root))
    {
        cout << "The root is: " << root << endl;
    }
    else
    {
        cout << "Failed in finding a root." << endl;
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