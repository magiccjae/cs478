
/*
	Author: Jae Lee
*/


import java.util.Random;
import java.util.ArrayList;

public class PerceptronLearner extends SupervisedLearner {

	double[] m_labels;
	double[] weights_two;
	double[][] weights_more;
	int outputs;
	
	Random random;
	public PerceptronLearner(Random rand){
		random = rand;
	}
	

	public void train(Matrix features, Matrix labels) throws Exception {
//		m_labels = new double[labels.cols()];
				
		// trying to figure out how many outputs there can be
		System.out.println("The number of outputs: " + labels.valueCount(0));		
		outputs = labels.valueCount(0);	// this determines the number of rows of weight
		double learning_rate = 0.1;
			
		if(outputs > 2){
			int weight_col = features.cols()+1;
			weights_more = new double[outputs][weight_col];
			
			// initialize the weights_more
			for(int i = 0; i < outputs; i++){
				for(int j = 0; j < weight_col; j++){
					weights_more[i][j] = random.nextDouble();
				}
			}
			
			boolean condition = true;
			int iteration = 0;
			ArrayList<Double> accuracy_list = new ArrayList<Double>();		
			int initial_iteration = 20;
			while(condition){
				
				System.out.println(iteration + " th iteration");
				for(int i = 0; i < outputs; i++){
					for(int j = 0; j < features.rows(); j++){
						double net = -1*weights_more[i][0];

						for(int k = 0; k < features.cols(); k++){
							net += features.get(j, k)*weights_more[i][k+1];
						}
						int output = 0;
						if(net > 0){
							output = 1;
						}
						else{
							output = 0;
						}
						int target = 0;
						if(labels.get(j, 0) == i){
							target = 1;
						}
						else{
							target = 0;
						}
						
//						System.out.println("answers: " + labels.get(j, 0) + "     " + "which perceptron: " + i + "     " + "output: " + output  + "     " + "target: " + target);
						
						for(int k = 0; k < weight_col; k++){
							if(k == 0){
								weights_more[i][k] = weights_more[i][k]-learning_rate*(output-target)*(-1);
							}
							else{
								weights_more[i][k] = weights_more[i][k]-learning_rate*(output-target)*(features.get(j, k-1));
							}
						}
					}
					
				}
				double current_accuracy = calculate_accuracy(features, labels);
				accuracy_list.add(current_accuracy);
				
				if(iteration >= initial_iteration){
					condition = continue_or_not(accuracy_list, current_accuracy, iteration);
				}
				iteration ++;
				
				features.shuffle(random, labels);
			}

		}
		else if(outputs == 2){
			int weight_col = features.cols()+1;
			weights_two = new double[weight_col];
			for(int i = 0; i < weight_col; i++){
				weights_two[i] = random.nextDouble();
			}

			boolean condition = true;
			int iteration = 0;
			ArrayList<Double> accuracy_list = new ArrayList<Double>();		
			int initial_iteration = 20;

			while(condition){
				System.out.println(iteration + " th iteration");
				
				for(int i = 0; i < features.rows(); i++){

					double net = -1*weights_two[0];
					for(int j = 0; j < features.cols(); j++){
						net += features.get(i, j)*weights_two[j+1];	
					}
					int output = 0;
					if(net > 0){
						output = 1;
					}
					else{
						output = 0;
					}
					for(int j = 0; j < weight_col; j++){
						if(j == 0){
							weights_two[j] = weights_two[j]-learning_rate*(output-labels.get(i, 0))*(-1);
						}
						else{
							weights_two[j] = weights_two[j]-learning_rate*(output-labels.get(i, 0))*(features.get(i, j-1));
						}
					}
				}
				double current_accuracy = calculate_accuracy(features, labels);
				accuracy_list.add(current_accuracy);
				
				if(iteration >= initial_iteration){
					condition = continue_or_not(accuracy_list, current_accuracy, iteration);
				}
				iteration ++;
				features.shuffle(random, labels);
			}
			
//			// weight print out
//			for(int i = 0; i < weights_two.length; i++){
//				System.out.println(i + " th weight: " + weights_two[i]);
//			}
				
		}
		
	}

	public void predict(double[] features, double[] labels) throws Exception {

		if(outputs > 2){
			
			double[] net = new double[outputs];
			for(int i = 0; i < outputs; i++){
				net[i] = -1*weights_more[i][0];
				for(int j = 0; j < features.length; j++){
					net[i] += features[j]*weights_more[i][j+1];
				}
			}
			
			double max = net[0];
			double index = 0;
			for(int i = 0; i < outputs; i++){
				if(net[i] > max){
					max = net[i];
					index = i;
				}
			}
			labels[0] = index;
		}
		else if(outputs == 2){
			double net = -1*weights_two[0];
			for(int i = 0; i < features.length; i++){
				net += features[i]*weights_two[i+1];			
			}			
			if(net > 0){
				labels[0] = 1;
			}
			else{
				labels[0] = 0;
			}
		}
	}
	
	public boolean continue_or_not(ArrayList<Double> accuracy_list, double current_accuracy, int iteration){
		
		double sum = 0;
		int total = 20;
		for(int i = 0; i < total; i++){
			sum += accuracy_list.get(iteration-total+i);
		}
		double average = sum/total;
		if(current_accuracy-average < 0.01){
			return false;
		}
		else{
			return true;
		}
		
	}
	
	public double calculate_accuracy(Matrix features, Matrix labels) throws Exception{
		int correctCount = 0;
		double[] prediction = new double[1];
		for(int i = 0; i < features.rows(); i++)
		{
			double[] feat = new double[features.cols()];
			feat = features.row(i);
								
			int targ = (int)labels.get(i, 0);
			predict(feat, prediction);
			int pred = (int)prediction[0];
								
			if(pred == targ)
				correctCount++;
		}
		double current_accuracy = (double)correctCount / features.rows();
		System.out.println("accuracy: " + current_accuracy);
		
		return current_accuracy;
	}


	@Override
	public double predict(double[] features, int target) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

}
