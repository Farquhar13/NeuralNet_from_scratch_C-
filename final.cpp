/*
weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

Run on Windows 10 in Visual Studio Code
AEP 4380 
Author: Collin Farquhar
*/

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <ctime>
#include "matrix.hpp"
#include <vector> // STD vector class

typedef arrayt<double> mdoub;

// hyper parameters
double leak = 0.5, alpha = 0.001;
const int n_input = 10, n_hidden_layers = 1, n_hidden_nodes = 5, n_out_nodes = 1;

// Declare weights and initialize biases globally for convience
mdoub w0(n_input+1, n_hidden_nodes); // +1 for bias
mdoub w1(n_hidden_nodes+1, n_out_nodes);
double b0 = 1.0, b1= 1.0; // biases

// keep track of MSE as the network trains
vector<double> mse_tracker;


void prepocess(mdoub& xTr, mdoub& yTr, mdoub& xTe, mdoub& yTe)
{
    /*
    Inputs:
        xTr: Training data
        yTr: Training labels
        xTe: Testing data
        yTe: Testing labels
    Desciption:
        Reads in csv file of training data, modifies xTr as matrix of data
        Reads in csv file of labels, modifies yTr as array of labels

    Note: Inputs should already be the shape of the respective csv file
    */

    // xTr
    ifstream infile( "x_10k.txt" );

    int count = 0, max = xTr.n1();
    while (infile)
    {
        string s;
        if (!getline( infile, s )) break;
        if (count == -1) continue;

        istringstream ss( s );

        int idx = 0;
        while (ss)
        {
            string s;
            if (!getline( ss, s, ',' )) break;
            //cout << s << endl;
            xTr(count, idx) = stod(s); // stod -> string to double
            idx += 1;
        }
        count += 1;
        if (count == max) break;
    }
    infile.close();


    // yTr
    ifstream yfile( "y_10k.txt" );

    count = 0, max = yTr.n1();
    while (yfile)
    {
        string s;
        if (!getline( yfile, s )) break;

        yTr(count) = stod(s); // stod -> string to double

        count += 1;
        if (count == max) break;
    }
    yfile.close();
}

inline double myrand(unsigned int &iseed)
{
    /*
        
        Citation:
            Dr. Kirkland, Cornell, AEP 4380 homework 9
            Numerical Recipies 3rd edition p. 356
        Description:
            returns a random number between 0-1 using lcg modulo 2^32 
    */
    const static unsigned int a=1372383749ul, c=1289706101ul;
    const static double m=4294967296.0; 

    iseed = a*iseed + c;
    return( ((double) iseed)/m);
}

double leaky_ReLU(double z)
{
    // activation function
    if (z > 0){
        return z;
    }
    else{
        return leak*z; // lr is a hyperparamater 
    }
}

double leaky_ReLU_deriv(double z)
{
    // derivative of activation function
    if (z > 0){
        return 1;
    }
    else{
        return leak; // lr is a hyperparamater 
    }
}

/*
double sigmoid(double z)
{
    // final layer activation
    return(1/(1+exp(-z)));
}

double sigmoid_deriv(double z)
{
    return(sigmoid(z)*(1-sigmoid(z)));
}
*/

mdoub add_bias(mdoub a, double bias)
{
    if (a.n2() != 1) cout << "you should only add bias to a vector" << endl;

    const int n_rows = a.n1();
    mdoub ab(n_rows+1, 1);
    for(int i=0; i < n_rows+1; i++)
    {
        if (i == n_rows) ab(i) = bias;
        else ab(i) = a(i);
    }
    return ab;
}

mdoub forward_prop(mdoub input, double (*layer_f)(double))
{
    mdoub inputb = add_bias(input, b0);
    // H is vector of hidden layer activations of weighted input sums
    mdoub w0T = transpose(w0);
    mdoub H = applyFunction(layer_f, dot(w0T, inputb));

    // add bias to hidden layer
    mdoub Hb = add_bias(H, b1);

    cout << "H = " << Hb.n1() << " x " << Hb.n2() << endl;
    cout << "w1 = " << w1.n1() << " x " << w1.n2() << endl;
    mdoub w1T = transpose(w1);
    mdoub Y = dot(w1T, Hb);
    return Y;
}

double mse(double pred, double y)
{
    /* 
        Loss Function: Mean Squared Error
        Input:
            pred: predicition of network
            y: true value
        Output:
            Computed MSE
    */
    return 0.5*(pred - y)*(pred - y); //using a factor of 1/2 to cancel with derivative
}


int main()
{
    // goal: load ruby data
    // also, try to focus :)
    mdoub xTr(10000,10);
    mdoub yTr(10000);
    mdoub xTe(2000,10); // not sure if will use
    mdoub yTe(2000);

    prepocess(xTr, yTr, xTe, yTe);
    /*
    cout << "after preprocess" << endl;
    for(int i=0; i<xTr.n2(); i++)
    {
        cout << xTr(0,i) << endl;
    }

    cout << "y" << endl;
    for(int i=9999; i>9989; i--)
    {
        cout << yTr(i) << endl;
    }
    */

    // Randomize weights
    unsigned int seed = time(NULL);

    for(int i=0; i < w0.n1(); i++)
    {
        for(int j=0; j < w0.n2(); j++)
        {
            w0(i,j) = myrand(seed)-0.5; // -0.5 to center mean at 0
        }
    }
    //print(w0);
    for(int i=0; i < w1.n1(); i++)
    {
        for(int j=0; j < w1.n2(); j++)
        {
            w1(i,j) = myrand(seed) -0.5; // -0.5 to center mean at 0
        }
    }
    //print(w1);

    // feedforward (first try with one, eventually loop)
    // get single training example
    mdoub example(xTr.n2(),1);
    for (int j=0; j < example.n1(); j++){
        example(j) = xTr(0,j);
    }
    //print(example);
    //mdoub ex_in = add_bias(example, b0);
    double ex_y = yTr(0,0);
    //print(ex_in);
    

    /*
    // try forward propagation 
    mdoub Y = forward_prop(example, leaky_ReLU);
    if (Y.n() > 1) cout << "There should only be one output node." << endl;
    double pred = Y(0,0); // can convert back to double because just one output node 
    cout << "pred = " << pred << endl;
    cout << "ex_y = " << ex_y << endl;
    double mse_err = mse(pred, ex_y);
    cout << mse_err << endl;
    */


    // ------------------   forward prop in main -----------------------------
    mdoub inputb = add_bias(example, b0);

    // H is vector of hidden layer activations of weighted input sums
    mdoub w0T = transpose(w0);

    mdoub in_h = dot(w0T, inputb);
    mdoub H = applyFunction(leaky_ReLU, in_h);

    // add bias to hidden layer
    mdoub Hb = add_bias(H, b1);

    mdoub w1T = transpose(w1);
    mdoub Y = dot(w1T, Hb);
    double pred = Y(0,0); // can convert back to double because just one output node 
    cout << "outcome of forward prop = " << pred << endl;
    
    // back prop
    // update weights going into output
    double delta = pred - ex_y;

    // update w1
    mdoub w1_grad = delta*Hb; // overloaded * operator (scalar multiplication)
    cout << "w0 rows = " << w0.n1() << endl;
    cout << "w0 columns = " << w0.n2() << endl;
    
    w1_grad = alpha*w1_grad; // (scalar multiplication)
    w1 = w1 - w1_grad;
    
    // update w0
    mdoub w0_grad (w0.n1(), w0.n2());

    for(int i=0; i < w0.n1(); i++)
    {
        for(int j=0; j < w0.n2();j++)
        {
            w0_grad(i,j) = delta * leaky_ReLU_deriv(in_h(j,0)) * inputb(i,0);
        }
    }
    w0_grad = alpha*w0_grad; // (scalar multiplication)
    w0 = w0 - w0_grad;
    
    cout << "w0 rows = " << w0.n1() << endl;
    cout << "w0 columns = " << w0.n2() << endl;
    // I think the weights have been updated, let's check
    print(w0);
    // time to loop dis 
    return(EXIT_SUCCESS); 
}