import java.util.ArrayList;
import java.util.Collections;


public class InstanceBasedLearner extends SupervisedLearner{

	public class Node implements Comparable{
		
		double distance;
		double weight;
		int index;
		double target;
		
		public Node(double distance, double weight, int index, double target){
			this.distance = distance;
			this.weight = weight;
			this.index = index;
			this.target = target;
		}
		public Node() {
			// TODO Auto-generated constructor stub
		}

		public double get_distance(){
			return distance;
		}
		public double get_target(){
			return target;
		}
		public double get_weight(){
			if(distance == 0){
				return Integer.MAX_VALUE;
			}
			weight = 1.0/Math.pow(distance,2);
			return weight;
		}
		
		@Override
		public int compareTo(Object o) {
			// TODO Auto-generated method stub
			Node other = (Node) o;
			if(distance < other.get_distance()){
				return -1;
			}
			else if(distance > other.get_distance()){
				return 1;
			}
			else{
				return 0;
			}
		}
		
	}
	
	Matrix global_feature;
	Matrix global_labels;
	Matrix reduction_feature;
	Matrix reduction_labels;
	ArrayList<Integer> drop = new ArrayList<>();
	
	public InstanceBasedLearner(){
		
	}
	
	public double calculate_accuracy(Matrix features, Matrix labels) throws Exception{
		int correct = 0;
		double[] prediction = new double[1];
		for(int i = 0; i < features.rows(); i++){
			double[] instance = features.row(i);
			int target = (int) labels.get(i, 0);
			predict2(instance, prediction);
			int predict = (int) prediction[0];
			if(predict == target){
				correct++;
			}
		}
		return (double)correct/features.rows();
	}
	public void predict2(double[] features, double[] labels) throws Exception {
		int k_value = 3;
		ArrayList<Node> neighbor = new ArrayList<>();
		
		int operation = 0;
		if(operation == 0){
			// knn without weight
			for(int i = 0; i < reduction_feature.rows(); i++){
				if(!drop.contains(i)){
					double distance = 0;
					for(int j = 0; j < features.length; j++){
						//					if((global_feature.get(i, j) == Matrix.MISSING) || (features[j] == Matrix.MISSING)){
						//						distance++;
						//					}else{
						distance += Math.pow((reduction_feature.get(i, j)-features[j]),2);
						//					}
					}
					distance = Math.sqrt(distance);
					Node new_node = new Node(distance, 0, i, reduction_labels.get(i, 0));
					//				add_node(new_node, neighbor, k_value);
					neighbor.add(new_node);
				}
			}
			sort_neighbor(neighbor);
			labels[0] = vote(neighbor, k_value, 0);
		}
	}	
	public void train(Matrix features, Matrix labels) throws Exception {
		
		double train_percent = 1;
		int train_size = (int)(train_percent*features.rows());
		Matrix train_features = new Matrix(features, 0, 0, train_size, features.cols());
		Matrix train_labels = new Matrix(labels, 0, 0, train_size, 1);
		Matrix test_features = new Matrix(features, train_size, 0, features.rows()-train_size, features.cols());
		Matrix test_labels = new Matrix(labels, train_size, 0, features.rows()-train_size, 1);

		global_feature = train_features;
		global_labels = train_labels;

		reduction_feature = train_features;
		reduction_labels = train_labels;
		
//		System.out.println("Initial instances: " + train_features.rows());
//		System.out.println("Initial accuracy: " + calculate_accuracy(test_features, test_labels));
//		
//		for(int i = 0; i < train_features.rows(); i++){
//			
//			double prev_accuracy = calculate_accuracy(test_features, test_labels);
//			
//			drop.add(i);
//			
//			double current_accuracy = calculate_accuracy(test_features, test_labels);
//			
//			if(current_accuracy < prev_accuracy){
//				drop.remove(drop.size()-1);
//			}
//			
//		}
//		System.out.println("Final instances: " + (train_features.rows()- drop.size()));
//		System.out.println("Final accuracy: " + calculate_accuracy(test_features, test_labels));

	}
	
	@SuppressWarnings("unchecked")
	public void sort_neighbor(ArrayList<Node> neighbor){
		Collections.sort(neighbor);
	}
	
//	public void add_node(Node new_node, ArrayList<Node> neighbor, int k_value){
//		if(neighbor.size() < k_value){	// add
//			neighbor.add(new_node);
//			sort_neighbor(neighbor);
//		}
//		else{ 		// if distance is closer, replace
//			double last_distance = neighbor.get(neighbor.size()-1).get_distance();
//			double new_distance = new_node.get_distance();
//			if(last_distance > new_distance){
//				neighbor.remove(neighbor.remove(neighbor.size()-1));
//				neighbor.add(new_node);
//			}
//			sort_neighbor(neighbor);
//		}
//		
//	}
	
	public double vote(ArrayList<Node> neighbor, int k_value, int operation){
		
		int[] vote_array = new int[global_labels.valueCount(0)];
		if(operation == 0){
			for(int i = 0; i < k_value; i++){
				vote_array[(int)neighbor.get(i).get_target()] += 1;
			}
		}
		else if(operation ==1){
			for(int i = 0; i < k_value; i++){
				vote_array[(int)neighbor.get(i).get_target()] += 1*neighbor.get(i).get_weight();
			}
		}
		
		int index = 0;
		int max = 0;
		for(int i = 0; i < vote_array.length; i++){
			if(vote_array[i] > max){
				index = i;
				max = vote_array[i];
			}
		}
		return (double)index;
	}
	
	public void predict(double[] features, double[] labels) throws Exception {
		int k_value = 3;
				
		ArrayList<Node> neighbor = new ArrayList<>();
		
		int operation = -1;
		if(operation == -1){ 		// knn for the group project
			for(int i = 0; i < global_feature.rows(); i++){
				double distance = 0;
				for(int j = 0; j < features.length; j+=2){
					double x_diff = global_feature.get(i, j) - features[j];
					double y_diff = global_feature.get(i, j+1) - features[j+1];
					distance += Math.sqrt(Math.pow(x_diff, 2) + Math.pow(y_diff, 2));			
				}
				Node new_node = new Node(distance, 0, i, global_labels.get(i, 0));
				neighbor.add(new_node);
			}
			sort_neighbor(neighbor);
			labels[0] = vote(neighbor, k_value, 0);
			
		}
		else if(operation == 0){
			// knn without weight
			for(int i = 0; i < global_feature.rows(); i++){
				double distance = 0;
				for(int j = 0; j < features.length; j++){
//					if((global_feature.get(i, j) == Matrix.MISSING) || (features[j] == Matrix.MISSING)){
//						distance++;
//					}else{
						distance += Math.pow((global_feature.get(i, j)-features[j]),2);
//					}
				}
				distance = Math.sqrt(distance);
				Node new_node = new Node(distance, 0, i, global_labels.get(i, 0));
				neighbor.add(new_node);
			}
			sort_neighbor(neighbor);
			labels[0] = vote(neighbor, k_value, 0);
		}
		else if(operation == 1){
			// knn with weight
			for(int i = 0; i < global_feature.rows(); i++){
				double distance = 0;
				for(int j = 0; j < features.length; j++){
					if((global_feature.get(i, j) == Matrix.MISSING) || (features[j] == Matrix.MISSING)){
						distance++;
					}else{
						distance += Math.pow((global_feature.get(i, j)-features[j]),2);
					}
				}
				distance = Math.sqrt(distance);
				Node new_node = new Node(distance, 0, i, global_labels.get(i, 0));
				neighbor.add(new_node);
			}
			sort_neighbor(neighbor);
			labels[0] = vote(neighbor, k_value, 1);
			
		}
		else if(operation == 2){
			// regression without weight
			for(int i = 0; i < global_feature.rows(); i++){
				double distance = 0;
				for(int j = 0; j < features.length; j++){
					distance += Math.pow((global_feature.get(i, j)-features[j]),2);
				}
				distance = Math.sqrt(distance);
				Node new_node = new Node(distance, 0, i, global_labels.get(i, 0));
				neighbor.add(new_node);
			}
			sort_neighbor(neighbor);
			double target_sum = 0;
			for(int i = 0; i < k_value; i++){
				target_sum += neighbor.get(i).get_target();
			}
			labels[0] = target_sum/(double)k_value;
			
		}
		else if(operation == 3){
			// regression with weight
			for(int i = 0; i < global_feature.rows(); i++){
				double distance = 0;
				for(int j = 0; j < features.length; j++){
					distance += Math.pow((global_feature.get(i, j)-features[j]),2);
				}
				distance = Math.sqrt(distance);
				Node new_node = new Node(distance, 0, i, global_labels.get(i, 0));
				neighbor.add(new_node);
			}
			sort_neighbor(neighbor);
			double target_sum = 0;
			double weight_sum = 0;
			for(int i = 0; i < k_value; i++){
				target_sum += neighbor.get(i).get_target()*neighbor.get(i).get_weight();
				weight_sum += neighbor.get(i).get_weight();
			}
			labels[0] = target_sum/weight_sum;
			
		}
		
		
	}

	@Override
	public double predict(double[] features, int target) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

}
