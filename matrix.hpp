/*
matrix.hpp

Implements linear algebra operations for arrayt<double> vectors and matrices
needed for matrix based neural networks. 

Functions:
    dot:
        matrix multiplication
        matrix vector multiplication
        vector vector dot product
    transpose:
        transposes matrix or vector
    multiply:
        element-wise mutliplication of matrices or vectors
    applyFunction:
        applies a function to each element of matrix or vector
    Overwrites '-' for matrices and vectors:
        element-wise subtraction
    Overwrites '+' for matrices and vectors:
        element-wise addition
    Overwrites '*' for double and matrix/vector:
        scalar multiplication
    print:
        outputs matrix or vector
    
Run on Windows 10 in Visual Studio Code
AEP 4380 
Author: Collin Farquhar
*/

#ifndef MATRIX
#define MATRIX

#include <cstdlib>
#include <iostream> //  stream IO
#include <iomanip>  //  to format the output
#include "arrayt.hpp" // Author: Dr. Kirkland, Cornell

using namespace std;

arrayt<double> dot(arrayt<double>& a, arrayt<double>& b)
{
    /*
    Returns the product of matrices or vectors
    Inputs: 
        a: array<double> matrix of size a_r, a_c
        b: array<double> matrix of size b_r x b_c
    Output:
        product: array<double> of the resulting matrix/vector after matrix 
        mulitiplication of a and b
    Description:
        If a and b are matrices, perform matrix multiplication, product is matrix
        If a is a matrix and b is a vector, perform transformation, product is vector
        If transpose(a) and b are vectors, dot product, product is a 1x1 (scalar)
    */

    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();
    const int b_s = b.n(), b_r = b.n1(), b_c = b.n2();

    if (a_c != b_r){
        cout << "dot product dimensions do not match" << endl;
        //exit(EXIT_FAILURE); // uncomment if you'd like the program to stop
    }
    arrayt<double> product(a_r, b_c);

    for(int i=0; i < a_r; i++)
    {
        for(int j=0; j < b_c; j++)
        {
            product(i, j) = 0.0;
            for(int k=0; k < a_c; k++)
            {
                product(i,j) += a(i, k)*b(k, j);
            }
        }
    }
    return product;
}

arrayt<double> transpose(arrayt<double>& x)
{
    /*
    Returns the transpose of x
    Input: arrayt<double> matrix x
    Output: arrayt<double> matrix xT
    Description: 
        Takes transpose of x and stores it in a xT, flips rows and columns
    */
    const int r = x.n1(), c = x.n2();
    arrayt<double> xT(c,r);

    for(int i=0; i < r; i++)
    {
        for(int j=0; j < c; j++)
        {
            xT(j,i) = x(i,j);
        }
    }
    return xT;
}

arrayt<double> multiply(arrayt<double>& a, arrayt<double>& b){
    
    /*  
    Inputs:
        a: arrayt<double> vector (could be matrix)
        b: arrayt<double> vector (could be matrix)
    Output: arrayt<double> product of element-wise multiplication
    Description:
        Returns the result of elementwise multiplication on two vectors 
        of the same dimensionality, will also work for 2 matrices of the same shape
    */
    
    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();
    const int b_s = b.n(), b_r = b.n1(), b_c = b.n2();

    if (a_r != b_r || a_c != b_c){
        cout << "vectors must be the same size to multiply" << endl;
        //exit(EXIT_FAILURE); // uncomment if you'd like the program to stop
    } 

    arrayt<double> product(a_r, b_c);
    
    for(int i = 0; i < a_r; i++){
        for(int j = 0; j < a_c; j++)
        {
            product(i,j) = a(i,j)*b(i,j);
        }
    }
    
    return product;
}

arrayt<double> operator-(arrayt<double>& a, arrayt<double>& b)
{ 
    /*  
    Inputs:
        a: arrayt<double> vector (could be matrix)
        b: arrayt<double> vector (could be matrix)
    Output: arrayt<double> difference of element-wise subtraction
    Description:
        overload the c++ '-' operator. When using '-' on vectors or matrices, 
        returns the result of element-wise subtraction
    */
    
    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();
    const int b_s = b.n(), b_r = b.n1(), b_c = b.n2();

    if (a_r != b_r || a_c != b_c){
        cout << "vectors must be the same size to subtract" << endl;
        //exit(EXIT_FAILURE); // uncomment if you'd like the program to stop
    } 

    arrayt<double> difference(a_r, b_c);
    
    for(int i = 0; i < a_r; i++){
        for(int j = 0; j < a_c; j++)
        {
            difference(i,j) = a(i,j)-b(i,j);
        }
    }
    
    return difference;
}

arrayt<double> operator+(arrayt<double>& a, arrayt<double>& b)
{ 
    /*  
    Inputs:
        a: arrayt<double> vector (could be matrix)
        b: arrayt<double> vector (could be matrix)
    Output: arrayt<double> difference of element-wise subtraction
    Description:
        overload the c++ '+' operator. When using '+' on vectors or matrices, 
        returns the result of element-wise addition
    */
    
    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();
    const int b_s = b.n(), b_r = b.n1(), b_c = b.n2();

    if (a_r != b_r || a_c != b_c){
        cout << "vectors must be the same size to add" << endl;
        //exit(EXIT_FAILURE); // uncomment if you'd like the program to stop
    } 

    arrayt<double> summed(a_r, b_c);
    
    for(int i = 0; i < a_r; i++){
        for(int j = 0; j < a_c; j++)
        {
            summed(i,j) = a(i,j)+b(i,j);
        }
    }
    
    return summed;
}

arrayt<double> operator*(double s, arrayt<double>& a)
{ 
    /*  
    Inputs:
        s: scalar double
        a: arrayt<double> vector (could be matrix)
    Output: arrayt<double> prod
    Description:
        overload the c++ '*' operator to allow for scalar * matrix and
        scalar * vector multiplication
    */
    
    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();

    arrayt<double> prod(a_r, a_c);
    
    for(int i = 0; i < a_r; i++){
        for(int j = 0; j < a_c; j++)
        {
            prod(i,j) = s*a(i,j);
        }
    }
    
    return prod;
}

arrayt<double> applyFunction(double (*function)(double), arrayt<double> a)   
{
    /*  
    Inputs:
        a: arrayt<double> vector (could be matrix)
    Output: arrayt<double> f, same shape as a
    Description:
        The vector f is the result of applying a function element-wise to a
    */
    const int a_s = a.n(), a_r = a.n1(), a_c = a.n2();
 
    arrayt<double> f(a_r, a_c);
    for(int i = 0; i < a_r; i++){
        for(int j = 0; j < a_c; j++)
        {
            f(i,j) = function(a(i,j));
        }
    }
 
    return f;
}

void print(arrayt<double> m)
{
    // Prints an arrayt matrix or vector

    for(int i=0; i < m.n1(); i++)
    {
        for(int j=0; j < m.n2(); j++)
        {
            if (j!= m.n2()-1) cout << m(i, j) << setw(10); 
            else cout << m(i,j);
        }
        cout << endl;
    }
}

#endif