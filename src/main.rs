struct neuralNet{
	config:neuralNetConfig,
	wHidden :Vec<i64>,
	bHidden :Vec<i64>,
	wOut    :Vec<i64>,
	bOut    :Vec<i64>
}
struct neuralNetConfig{
	inputNeurons  :i64,
	outputNeurons :i64,
	hiddenNeurons :i64,
	numEpochs     :i64,
	learningRate  :f64
}
fn new_network(config:neuralNetConfig)->neuralNet{
    return neuralNet { config: config, wHidden: (), bHidden: (), wOut: (), bOut: () }
}
fn sigmoid
fn main() {
    println!("Hello, world!");
}
