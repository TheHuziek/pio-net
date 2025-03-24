extern crate nalgebra as na;
use na::DMatrix;
use rand::Rng;
use std::f64::consts::E;
#[derive(Debug)]
struct NeuralNet {
    config: NeuralNetConfig,
    w_hidden: DMatrix<f64>,
    b_hidden: DMatrix<f64>,
    w_out: DMatrix<f64>,
    b_out: DMatrix<f64>,
}
#[derive(Debug, Clone, Copy)]
struct NeuralNetConfig {
    input_neurons: i64,
    output_neurons: i64,
    hidden_neurons: i64,
    num_epochs: i64,
    learning_rate: f64,
}
fn new_network(config: NeuralNetConfig) -> NeuralNet {
    return NeuralNet {
        config: config,
        w_hidden: DMatrix::zeros(
            config.hidden_neurons as usize,
            config.input_neurons as usize,
        ),
        b_hidden: DMatrix::zeros(config.hidden_neurons as usize, 1),
        w_out: DMatrix::zeros(
            config.output_neurons as usize,
            config.hidden_neurons as usize,
        ),
        b_out: DMatrix::zeros(config.output_neurons as usize, 1),
    };
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}
fn sigmoid_prime(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}
//train trains a neural network using backpropagation. It takes in a neural network, a matrix of input data, and a matrix of output data. It returns the trained neural network.
fn train(nn: NeuralNet, input_data:DMatrix<f64>) -> NeuralNet {
    let  w_hidden = nn.w_hidden;
    let  b_hidden = nn.b_hidden;
    let  w_out = nn.w_out;
    let  b_out = nn.b_out;
    let  config = nn.config;

    let mut rng = rand::thread_rng();
	
    let w_hidden = DMatrix::from_vec(
        w_hidden.nrows(),
        w_hidden.ncols(),
        w_hidden.iter().map(|_|rng.random::<f64>()).rev().collect(),
    );
    let b_hidden = DMatrix::from_vec(
        b_hidden.nrows(),
        b_hidden.ncols(),
        b_hidden.iter().map(|_| rng.random::<f64>()).rev().collect(),
    );
    let w_out = DMatrix::from_vec(
        w_out.nrows(),
        w_out.ncols(),
        w_out.iter().map(|_| rng.random::<f64>()).rev().collect(),
    );
    let b_out: na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> = DMatrix::from_vec(
        b_out.nrows(),
        b_out.ncols(),
        b_out.iter().map(|_| rng.random::<f64>()).rev().collect(),
    );
    let (w_hidden, b_hidden, w_out, b_out) = backpropagate(w_hidden, b_hidden, w_out, b_out, config, input_data);
    
    return NeuralNet {
        config: config,
        w_hidden,
        b_hidden,
        w_out,
        b_out,
    };
}
fn backpropagate(
    mut w_hidden: DMatrix<f64>,
    mut b_hidden: DMatrix<f64>,
    mut w_out: DMatrix<f64>,
    mut b_out: DMatrix<f64>,
    config: NeuralNetConfig,
    input_data: DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {

    // Actualizar los pesos de salida
    let output_errors_prime = output_errors.component_mul(&final_outputs.map(sigmoid_prime));
    let delta_w_out = learning_rate * dot(&output_errors_prime, &hidden_outputs.transpose());
    w_out += delta_w_out;

    // Actualizar los pesos ocultos
    let hidden_errors_prime = hidden_errors.component_mul(&hidden_outputs.map(sigmoid_prime));
    let delta_w_hidden = learning_rate * dot(&hidden_errors_prime, &inputs.transpose());
    w_hidden += delta_w_hidden;

    (w_hidden, w_out)
    // Implement the backpropagation algorithm here
    // For now, just return the weights and biases as they are
    (w_hidden, b_hidden, w_out, b_out)
}
fn forward_propagate(
	w_hidden: DMatrix<f64>,
	b_hidden: DMatrix<f64>,
	w_out: DMatrix<f64>,
	b_out: DMatrix<f64>,
	input_data: DMatrix<f64>,
) -> DMatrix<f64> {

	let hidden_layer_input = dot(&w_hidden, &input_data) + &b_hidden;
	let hidden_layer_output = hidden_layer_input.map(sigmoid);
	let output_layer_input = dot(&w_out, &hidden_layer_output) + &b_out;
	let output = output_layer_input.map(sigmoid);
	output
}


fn dot(m: &DMatrix<f64>, n: &DMatrix<f64>) -> DMatrix<f64> {
    let result = m * n;
    result
}
fn main() {
	let nnetconf= NeuralNetConfig {
		input_neurons: 2,
		output_neurons: 1,
		hidden_neurons: 2,
		num_epochs: 1000,
		learning_rate: 0.1,
	};
	let nnet= new_network(nnetconf);
    let input_data = DMatrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
	let train_net=train(nnet, input_data);
	println!("{:?}", train_net);
}
