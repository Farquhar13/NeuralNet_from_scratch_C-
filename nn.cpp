/*
nn.cpp

Neural Network coded from scratch 
Predict redshift from SDSS data

Run on Windows 10 in Visual Studio Code
AEP 4380 Final Project 
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

#define ARRAYT_BOUNDS_CHECK
typedef arrayt<double> mdoub;


// hyper parameters
double leak = 0.5, alpha = 0.001, threshold = 1e-8;
const int n_input = 10, n_hidden_layers = 1, n_hidden_nodes = 5, n_out_nodes = 1;

// Declare weights and initialize biases globally for convience
mdoub w0(n_input+1, n_hidden_nodes); // +1 for bias
mdoub w1(n_hidden_nodes+1, n_out_nodes);
double b0 = 1.0, b1= 1.0; // biases

// keep track of MSE as the network trains
vector<double> mse_tracker;

// track predictions and actual values
vector<double> predictions;
vector<double> actual;

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
    ifstream infile( "x_prep.txt" );

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
    ifstream yfile( "y_prep.txt" );

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
    mdoub ab(n_rows+1,1);
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

void write_mse()
{
    ofstream f;
    f.open("mse.dat");
    for (int i=0; i<mse_tracker.size(); i++)
    {
        f << mse_tracker[i] << endl;
    }
    f.close();

    ofstream t;
    t.open("mse.txt");
    for (int i=0; i<mse_tracker.size(); i++)
    {
        t << mse_tracker[i] << endl;
    }
    t.close();
}

void checkw(mdoub w0, mdoub w1)
{
    for (int i=0; i < w0.n1(); i++)
    {
        for (int j=0; j < w0.n2(); j++)
        {
            if (isinf(w0(i,j))) cout << "infinity in w0" << endl;
        }
    }

    for (int i=0; i < w1.n1(); i++)
    {
        if (isinf(w1(i))) cout << "infinity in w1" << endl;
    }
}

bool stop(mdoub w0_grad, mdoub w1_grad)
{
    double max = 0.0;
    for (int i=0; i < w0_grad.n1(); i++)
    {
        for (int j=0; j < w0_grad.n2(); j++)
        {
            if (w0_grad(i,j) > max) max = w0_grad(i,j);
            if (isinf(w0_grad(i,j))) cout << "infinity in w0_grad" << endl;
        }
    }

    for (int i=0; i < w1_grad.n1(); i++)
    {
        if (w1_grad(i) > max) max = w1_grad(i);
        if (isinf(w1_grad(i))) cout << "infinity in w1_grad" << endl;
    }

    if (max < threshold) return true;
    else return false;
}

void weight_writer(mdoub w0, mdoub w1)
{
    ofstream w;
    w.open("weights.txt");
    w << "w0" << endl;
    for (int i=0; i < w0.n1(); i++)
    {
        for (int j=0; j < w0.n2(); j++)
        {
            w << w0(i,j) << endl;
        }
    }
    w << "w1" << endl;
    for (int i=0; i<w1.n1(); i++)
    {
        w << w1(i) << endl;
    }

    w.close();
}

void eval_performance(mdoub& xTr, mdoub& yTr)
{
    // get the last 100 points
    const int last = xTr.n1()-1;
    const int n_ex = 100;
    //vector<double> valid_mse;
    //vector<double> benchmark;
    double valid_sum = 0;
    double benchmark_sum = 0; 
    const double avg_redshift = 0.35960330678661007; // computed in python

    for (int i=last; i > (last-n_ex) ; i--)
    {
        // get x example
        mdoub example(xTr.n2(),1);
        for (int j=0; j < example.n1(); j++){
            example(j) = xTr(i,j);
        }

        // get y example
        double ex_y = yTr(i);

        // ----------------     foward prop         ---------------------
        // add bias to example for input into the network
            mdoub inputb = add_bias(example, b0); 

            // compute propogation of inputs to hidden layer
            mdoub w0T = transpose(w0);
            mdoub in_h = dot(w0T, inputb);

            // H is vector of hidden layer activations of weighted input sums
            mdoub H = applyFunction(leaky_ReLU, in_h);

            // add bias to hidden layer
            mdoub Hb = add_bias(H, b1);

            // computer propogation from hiddern layer to output
            mdoub w1T = transpose(w1);
            mdoub Y = dot(w1T, Hb);
            double pred = Y(0); // can convert back to double because just one output node

            // compute mse
            valid_sum += mse(pred, ex_y);
            benchmark_sum += mse(avg_redshift, ex_y);
    }
    cout << "validation mse = " << valid_sum/n_ex << endl;
    cout << "benchmark mse = " << benchmark_sum/n_ex << endl;
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


    // LOOP
    //for(int i=0; i < xTr.n1(); i++)
    for(int index=0; index < xTr.n1(); index++)
    {
        // get x example
        mdoub example(xTr.n2(),1);
        for (int j=0; j < example.n1(); j++){
            example(j) = xTr(index,j);
        }

        // get y example
        double ex_y = yTr(index);

        // ------------------   forward prop in main    -----------------------------

        // add bias to example for input into the network
        mdoub inputb = add_bias(example, b0); 

        // compute propogation of inputs to hidden layer
        mdoub w0T = transpose(w0);
        mdoub in_h = dot(w0T, inputb);

        // H is vector of hidden layer activations of weighted input sums
        mdoub H = applyFunction(leaky_ReLU, in_h);

        // add bias to hidden layer
        mdoub Hb = add_bias(H, b1);

        // computer propogation from hiddern layer to output
        mdoub w1T = transpose(w1);
        mdoub Y = dot(w1T, Hb);
        double pred = Y(0); // can convert back to double because just one output node 


        // ------------------    backprop in main    -----------------------------

        // calculate error of the predicition
        double delta = pred - ex_y;

        // update weights, going backwards from output

        // update w1
        mdoub w1_grad = delta*Hb; // overloaded * operator (scalar multiplication)
        w1_grad = alpha*w1_grad; // scalar multiplication of learning rate and gradient
        w1 = w1 - w1_grad; // update
        
        // update w0
        mdoub w0_grad (w0.n1(), w0.n2());

        for(int i=0; i < w0.n1(); i++)
        {
            for(int j=0; j < w0.n2();j++)
            {
                w0_grad(i,j) = delta * leaky_ReLU_deriv(in_h(j)) * inputb(i);
            }
        }
        w0_grad = alpha*w0_grad; // (scalar multiplication)
        w0 = w0 - w0_grad;

        // compute mse
        double ex_mse = mse(pred, ex_y);
        mse_tracker.push_back(ex_mse);

        //cout << ex_y << "   " << pred << "   " << ex_mse << endl;

        bool done = stop(w0_grad, w1_grad);
        if (done){
            cout << "stopping at iteration " << index << endl;
            print(w0);
            print(w1);
            break;
        }

        if (index ==xTr.n1()-1)
        {
            cout << "stopping at iteration " << index << endl;
            print(w0);
            print(w1);
            break;
        }

        /*
        cout << "iteration " << index << endl;
        cout << "w0_grad" << endl;
        print(w0_grad);
        cout << "w1_grad" << endl;
        print(w1_grad);
        cout << "mse = " << ex_mse << "\n" << endl;
        */

    }
    
    write_mse();

    eval_performance(xTr, yTr);

    return(EXIT_SUCCESS); 
}