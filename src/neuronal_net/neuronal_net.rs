extern crate nalgebra as na;
use na::{DMatrix, zero};
use rand::Rng;
use std::{clone, f64::consts::E, process::Output};
use types_of_activation::ActivationFunctions;
use crate::neuronal_net::{self, types_of_activation};
#[derive(Debug, Clone, Copy)]
pub struct Layer{
    number_of_neurons:u32,
    activation:ActivationFunctions,
}
#[derive(Debug, Clone)]
pub struct NeuralNet{
    config:NeuralNetConfig,
    pub input_layer:Layer,
    pub hidden_layers:Vec<Layer>,
    pub output_layer:Layer,
}

#[derive(Debug, Clone)]
pub struct NeuralNetCompiled {
    pub hidden_activations:Vec<ActivationFunctions>,
    pub hidden_weights:Vec<DMatrix<f64>>,
    pub hidden_biases:Vec<DMatrix<f64>>,
    pub output_activation:ActivationFunctions,
    pub output_weights:DMatrix<f64>,
    pub output_biases:DMatrix<f64>,
    config: NeuralNetConfig,
}
#[derive(Debug, Clone, Copy)]
pub struct NeuralNetConfig {
    pub num_epochs: i64,
    pub learning_rate: f64,
}
impl NeuralNet {
    pub fn new_layer(number_of_neurons:u32, activation:ActivationFunctions) -> Layer {
        return Layer {
            number_of_neurons: number_of_neurons,
            activation: activation,
        };
    }
    pub fn new_network(input_shape:usize,  hidden_layers:Vec<Layer>, output_shape:usize, config: NeuralNetConfig) -> NeuralNet {
        return NeuralNet {
            config: config,
            input_layer: Self::new_layer(input_shape as u32, ActivationFunctions::Sigmoid),
            hidden_layers: hidden_layers,
            output_layer: Self::new_layer(output_shape as u32, ActivationFunctions::Sigmoid),
        };
    }pub fn compile(&self) -> NeuralNetCompiled {
        let mut rng = rand::thread_rng();
        let input_neurons = self.input_layer.number_of_neurons as usize;

        let mut hidden_activations = Vec::new();
        let mut hidden_weights = Vec::new();
        let mut hidden_biases = Vec::new();

        // Inicialización de pesos y biases para capas ocultas
        let mut prev_neurons = input_neurons;
        
        for layer in &self.hidden_layers {
            let current_neurons = layer.number_of_neurons as usize;
            
            // Inicializar pesos con Xavier/Glorot initialization
            let bound = (6.0f64).sqrt() / ((prev_neurons + current_neurons) as f64).sqrt();
            let weights = DMatrix::from_fn(current_neurons, prev_neurons, |_, _| {
                rng.gen_range(-bound..bound)
            });
            
            // Inicializar biases en 0 o valores pequeños
            let biases = DMatrix::zeros(current_neurons, 1);
            
            hidden_weights.push(weights);
            hidden_biases.push(biases);
            hidden_activations.push(layer.activation);
            
            prev_neurons = current_neurons;
        }

        // Inicialización de pesos y biases para capa de salida
        let output_neurons = self.output_layer.number_of_neurons as usize;
        let bound = (6.0f64).sqrt() / ((prev_neurons + output_neurons) as f64).sqrt();
        let output_weights = DMatrix::from_fn(output_neurons, prev_neurons, |_, _| {
            rng.gen_range(-bound..bound)
        });
        let output_biases = DMatrix::zeros(output_neurons, 1);

        NeuralNetCompiled {
            hidden_activations,
            hidden_weights,
            hidden_biases,
            output_activation: self.output_layer.activation,
            output_weights,
            output_biases,
            config: self.config,
        }
    }
}
impl NeuralNetCompiled{
    pub fn train(){}
}
fn multiply(m: &DMatrix<f64>, n: &DMatrix<f64>) -> DMatrix<f64> {
    // Verificación de dimensiones (opcional, pero recomendado)
    assert_eq!(
        m.shape(),
        n.shape(),
        "Las matrices deben tener las mismas dimensiones"
    );

    // Multiplicación elemento por elemento
    m.component_mul(n)
}
fn dot(m: &DMatrix<f64>, n: &DMatrix<f64>) -> DMatrix<f64> {
    // Verificación de dimensiones compatibles (opcional)
    assert_eq!(
        m.ncols(),
        n.nrows(),
        "El número de columnas de m debe coincidir con las filas de n"
    );

    // Multiplicación de matrices
    m * n // Equivalente a m.multiply(n)
}
fn subtract(m: &DMatrix<f64>, n: &DMatrix<f64>) -> DMatrix<f64> {
    m - n
}
fn add(m: &DMatrix<f64>, n: &DMatrix<f64>) -> DMatrix<f64> {
    m + n
}
fn scale(s: f64, m: &DMatrix<f64>) -> DMatrix<f64> {
    s * m
}
