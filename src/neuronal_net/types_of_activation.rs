use std::f64::consts::E;
#[derive(Debug,Clone, Copy)]
pub enum ActivationFunctions{
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
}
pub struct ActivationFunction{}   
impl ActivationFunctions {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunctions::Sigmoid => sigmoid(x),
            ActivationFunctions::ReLU => x.max(0.0),
            ActivationFunctions::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            ActivationFunctions::Tanh => x.tanh(),
            ActivationFunctions::Softmax => x.exp(), // Nota: Softmax se aplica a vectores
        }
    }
    
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunctions::Sigmoid => sigmoid_prime(x),
            ActivationFunctions::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunctions::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            ActivationFunctions::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunctions::Softmax => 1.0, // La derivada de softmax es más compleja
        }
    }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}
fn sigmoid_prime(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}
