extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;
use std::{clone, f64::consts::E};
mod neuronal_net;
use crate::neuronal_net::{
    neuronal_net::{ NeuralNet,NeuralNetConfig,NeuralNetCompiled},
    types_of_activation::ActivationFunctions
};
fn main() {    let config = NeuralNetConfig {
        num_epochs: 100,
        learning_rate: 0.01,
    };
    
    let hidden_layers = vec![
        NeuralNet::new_layer(16, ActivationFunctions::ReLU),
        NeuralNet::new_layer(8, ActivationFunctions::ReLU),
    ];
    
    let network = NeuralNet::new_network(
        4,          // neuronas de entrada
        hidden_layers,
        2,          // neuronas de salida
        config
    );
    
    let compiled_network = network.compile();
    println!("Red compilada: {:#?}", compiled_network);
}
