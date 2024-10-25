#include<bits/stdc++.h>
using namespace std;

// Solution of Linear Equations Functions

void jacobiIterativeMethod() {
    cout << "Jacobi iterative method code here" << endl;
}

void gaussSeidelIterativeMethod() {
    cout << "Gauss-Seidel iterative method code here" << endl;
}

void gaussEliminationMethod() {
    cout << "Gauss Elimination method code here" << endl;
}

void gaussJordanEliminationMethod() {
    cout << "Gauss-Jordan Elimination method code here" << endl;
}

void LUFactorizationMethod() {
    cout << "LU factorization method code here" << endl;
}

// Solution of Non-linear Equations functions

void biSectionMethod() {
    cout << "Bi-section method code here" << endl;
}

void falsePositionMethod() {
    cout << "False position method code here" << endl;
}

void secantMethod() {
    cout << "Secant method code here" << endl;
}

void newtonRaphsonMethod() {
    cout << "Newton-Raphson method code here" << endl;
}


// Solution of Differential Equations Funtions

void rungeKuttaMethod() {
    cout << "Runge-Kutta Method code here" << endl;
}

// Matrix Inversion Functions 

void matrixInversion() {
    int n;
    cout << "Enter the order of the matrix: ";
    cin >> n;

    vector<vector<double>> a(n, vector<double>(n));

    cout << "Enter the matrix elements:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> a[i][j];
        }
    }

    vector<vector<double>> inverse(n, vector<double>(n, 0));

    // Initialize inverse as an identity matrix
    for (int i = 0; i < n; i++) {
        inverse[i][i] = 1;
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        double diag = a[i][i];
        for (int j = 0; j < n; j++) {
            a[i][j] /= diag;
            inverse[i][j] /= diag;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = a[k][i];
                for (int j = 0; j < n; j++) {
                    a[k][j] -= factor * a[i][j];
                    inverse[k][j] -= factor * inverse[i][j];
                }
            }
        }
    }

    // Display the inverse matrix
    cout << "Inverse matrix:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << inverse[i][j] << " ";
        }
        cout << endl;
    }
}

void menu() {
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

    switch(type_choice) {
        case 1:
            cout << "a. Jacobi iterative method." << endl;
            cout << "b. Gauss-Seidel iterative method." << endl;
            cout << "c. Gauss Elimination method." << endl;
            cout << "d. Gauss-Jordan Elimination method." << endl;
            cout << "e. LU factorization method." << endl;

            cout << "Enter your choice: ";
            cin >> method_choice; 

            switch(method_choice) {
                case 'a': jacobiIterativeMethod();
                    menu();
                    break;
                case 'b': gaussSeidelIterativeMethod();
                    menu();
                    break;
                case 'c': gaussEliminationMethod();
                    menu();
                    break;
                case 'd': gaussJordanEliminationMethod();
                    menu();
                    break;
                case 'e': LUFactorizationMethod();
                    menu();
                    break;
                default : 
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

            switch(method_choice) {
                case 'a': biSectionMethod();
                    menu();
                    break;
                case 'b': falsePositionMethod();
                    menu();
                    break;
                case 'c': secantMethod();
                    menu();
                    break;
                case 'd': newtonRaphsonMethod();
                    menu();
                    break;
                default : 
                    cout << "Choose correct option." << endl;
                    menu();
            }
            break;

        case 3: 
            cout << "a. Runge-Kutta Method" << endl;

            cout << "Enter your choice: ";
            cin >> method_choice;

            switch(method_choice) {
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

        default : 
            cout << "Choose correct option." << endl;
            menu();
    }

}

int main() {

    menu();

    return 0;
}