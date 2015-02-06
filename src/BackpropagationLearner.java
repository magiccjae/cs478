import java.util.Random;

public class BackpropagationLearner extends SupervisedLearner{
	
	double[] hidden_output;
	double[][] weight1;
	double[][] weight2;
	
	Random random;
	
	public BackpropagationLearner(Random rand){
		random = rand;
	}
	
	public double get_gaussian(){
		double min = -0.3;
		double max = 0.3;
		
		boolean condition = true;
		while(condition){
			double gaussian = random.nextGaussian();
			if(gaussian > max || gaussian < min || gaussian == 0){
				// don't do anything
			}
			else{
				return gaussian;
			}
		}
		return -1;
	}
	
	public void create_layer(int num_input, int num_output){
		
		int num_hidden_node = num_input*2;
		int weight1_row = num_input+1;
		int weight1_col = num_hidden_node;
		int weight2_row = num_hidden_node+1;
		int weight2_col = num_output;
		hidden_output = new double[num_hidden_node];
		weight1 = new double[weight1_row][weight1_col];
		weight2 = new double[weight2_row][weight2_col];
		
		for(int i = 0; i < weight1_row; i++){
			for(int j = 0; j < weight1_col; j++){
				weight1[i][j] = get_gaussian();
			}
		}
		for(int i = 0; i < weight2_row; i++){
			for(int j = 0; j < weight2_col; j++){
				weight2[i][j] = get_gaussian();
			}
		}
		
		
	}

	public void train(Matrix features, Matrix labels) throws Exception {
		int num_input = features.cols();
		int num_output = labels.valueCount(0);	
		create_layer(num_input, num_output);
	}

	public void predict(double[] features, double[] labels) throws Exception {
		
	}

}

