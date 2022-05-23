/*
* Fecha: 28/02/2022
* Autor: Santiago Javier Vivas Piamba
* Materia: HPC- 1
* Tema: Implementaciòn del algoritmo de regresiòn lineal
* Requerimientos:
* 1. Crear una clase que permita la manipulaciòn de los datos (extracciòn, normalizaciòn, entre otros) con Eigen.
* 2. Crear una clase que permita implementar el modelo o algoritmo de regresiòn lineal, con Eigen.
*/

#include "Extraccion/extraer.h"
#include "linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string.h>


int main(int argc, char *argv[]){

    /* Se crea un objeto del tipo extraer, para incluir los 3 argumentos que necesita el objeto. */
    extraer extraerData(argv[1], argv[2], argv[3]);

    //Se crea un objeto del tipo LinearRegresion, sin ningún argumento de entrada
    LinearRegression LR;

    // Se requiere probar la lectura del fichero y luego se requiere observar el dataset como un objeto de matriz tipo dataFrame
    std::vector<std::vector<std::string>> dataSET = extraerData.readCSV();

    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDATAF = extraerData.CSVtoEigen(dataSET, filas, columnas);
    std::cout<<"filas: "<<filas<<std::endl;
    std::cout<<"columnas: "<<columnas<<std::endl;


    /* Se imprime la matriz que contiene los datos del dataset */
    //std::cout<<" Se imprime el dataSet "<<std::endl<<MatrizDATAF<<std::endl;


    //std::cout<<"El promedio por columna es: "<<std::endl<<extraerData.promedio(MatrizDATAF)<<std::endl;
    //std::cout<<"La desviación estándar por columna es: "<<std::endl<<extraerData.desvEstandar(MatrizDATAF)<<std::endl;

    //Se crea la matriz para almacenar la normalización
    Eigen::MatrixXd matNormal = extraerData.Normalizador(MatrizDATAF);


    //se imprime el dataSet con datos normalizados
    //std::cout<<"Los datos normalizados son: "<<std::endl;
    //std::cout<<matNormal;

    //A continuación se dividen entrenamiento y prueba en conjuntos de datos de entrada (matNormal).

    Eigen::MatrixXd X_test, Y_test, X_train, Y_train;

    //Se dividen los datos y el 80% es para entrenamiento.
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> MatrixDividida = extraerData.trainTestSplit(matNormal, 0.8);
    //Se desempaqueta la tupla.
    std::tie(X_train, Y_train, X_test, Y_test) = MatrixDividida;
    //std::cout<< matNormal.rows()<<std::endl;
    //std::cout<<X_test.rows()<<std::endl;
    //std::cout<<X_train.rows()<<std::endl;

    //A continuacioń se hará el primer módulo de Machine Learning. Se hará una clase "RegresiónLineal". Con su correspondiente constructor de argumentos
    //de entrada y métodos para el cálculo del modelo RL. Se tiene en cuenta que el RL es un método estadístico, que define la relación entre las variables
    //independientes y la variable dependiente.
    //La idea principal, es definir una línea recta (Híper plano) con sus coeficientes (pendientes) y punto de corte.
    //Se tienen diferentes métodos para resolver RL. Para este caso se usará el método de los Mínimos Cuadrados Ordinarios. (OLS), por ser un
    //método sencillo y computacionalmente económico. Representa una solución óptima para conjunto de datos no complejos. El dataset a utilizar
    //es el de vinoRojo, el cuál tiene 11 variables (multivariable) independientes. Para ello hemos de implementar el algoritmo del gradiente descendiente,
    //cuyo objetivo principal es minimizar la función de costo.

    //Se define un vector para entrenamiento y para prueba inicializados en unos
    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    //Se redimensionan las matrices para ser ubicadas en el vector ed Unos: Similar al reshape() de numpy.
    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vectorTrain;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_test.cols()-1) = vectorTest;

    /* Se define el vector theta que se pasara al algoritmo del gradiente descendiente.
     *Básicamente es un vector de ceros del mismo tamaño del entrenamiento, adicionalmente
     * se pasará alpha y el número de iteraciones
     * */
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;
    /* A continuación se definen las variables de salida, que representan los coeficientes y el vector
     *de costo */
    Eigen::VectorXd thetaSalida;
    std::vector<float> costo;

    /*Se desempaqueta la tupla como objeto instanciando del gradiente descendiente.
     * */
    std::tuple<Eigen::VectorXd, std::vector<float>> objetoGradiente = LR.GradienteD(X_train,
                                                          Y_train, theta, alpha, iteraciones);
    std::tie(thetaSalida, costo) = objetoGradiente;

    //Se imprime los coeficientes para cada variable.
    //std::cout<<thetaSalida<<std::endl;

    //Se mprime para inspección ocular la función de costo
    /*for (auto var: costo) {
        std::cout<<var<<std::endl;
    }*/


    /* Se almacena la función de costo y las variables Theta a ficheros */

    //extraerData.vectorToFile(costo, "costo.txt");
//    extraerData.EigenToFile(thetaSalida, "theta.txt");


    /* Se calcula el promedio y la desviación estandar, para calcular las predicciones.
    Es decir, se debe de normalizar para calcular la métrica. */
    auto muData = extraerData.promedio(MatrizDATAF);
    auto muFeatures = muData(0, 11);
    auto escalado = MatrizDATAF.rowwise() - MatrizDATAF.colwise().mean();
    auto sigmaData = extraerData.desvEstandar(escalado);
    auto sigmaFeatures = sigmaData(0, 11);
    Eigen::MatrixXd y_train_hat = (X_train*thetaSalida*sigmaFeatures).array() + muFeatures;
    Eigen::MatrixXd y = MatrizDATAF.col(11).topRows(1279);
    float R2_score = extraerData.R2_score(y, y_train_hat);
    std::cout<<R2_score<<std::endl;

    extraerData.EigenToFile(y_train_hat, "y_train_hatCPP.txt");




















    return EXIT_SUCCESS;
}

