import java.util.ArrayList;


public class DecisionTree extends SupervisedLearner{

	public class Node{
		
		ArrayList<Integer> feature_index = new ArrayList<>();		
		ArrayList<Integer> label_index = new ArrayList<>();
		ArrayList<Node> children = new ArrayList<>();
		int feature_to_split;
		public Node(ArrayList<Integer> feature_index, ArrayList<Integer> label_index){
			this.feature_index = feature_index;			
			this.label_index = label_index;
			this.feature_to_split = -1;
		}
		
		
		public Node() {
			// TODO Auto-generated constructor stub
		}
		public int get_output(){
			int output = -1;
			int index = -1;
			int[] output_array = new int[global_labels.valueCount(0)];
			for(int i = 0; i < label_index.size(); i++){
				output_array[(int) global_labels.get(label_index.get(i), 0)] += 1;
			}
			for(int i = 0; i < output_array.length; i++){
				if(output_array[i] > output){
					output = output_array[i];
					index = i;
				}
			}
			return index;
		}
		public int get_feature_to_split(){
			return feature_to_split;
		}
		public void set_feature_to_split(int feature_to_split){
			this.feature_to_split = feature_to_split;
		}
		public ArrayList<Integer> get_feature_index(){
			return feature_index;
		}
		public ArrayList<Integer> get_label_index(){
			return label_index;
		}
		public ArrayList<Node> get_children(){
			return children;
		}
		public int total_instance(){
			return label_index.size();
		}
		
		public void add_to_children(Node new_node){
			children.add(new_node);
		}
		public void print_label_index(){
			for(int i = 0; i < label_index.size(); i++){
				System.out.print(label_index.get(i) + " ");
			}
			System.out.println();
		}
		public double get_info(){
			int[] num_instances = new int[global_labels.valueCount(0)];
			for(int i = 0; i < label_index.size(); i++){
//				num_instances[(int)labels.get(label_index.get(i),0)] = num_instances[(int)labels.get(label_index.get(i),0)]+1;
				num_instances[(int)global_labels.get(label_index.get(i),0)] += 1;

			}
			double info = 0;
			for(int i = 0; i < global_labels.valueCount(0); i++){
				if(num_instances[i] != 0){
					info += (double)num_instances[i]/(double)label_index.size()*(double)Math.log((double)num_instances[i]/label_index.size())/Math.log(2);
				}
			}
			
			return -info;
		}
		
	}
	
	Matrix global_features;
	Matrix global_labels;
	Node global_root;
	
	public DecisionTree(){
		global_root = new Node();
	}
	
	public void calculate_infoa(Node current_node){
		// Calculate infoa for all features
		ArrayList<Double> infoa_list = new ArrayList<>();
		boolean unknown_exist = false;
		boolean extra_node = false;
		for(int i = 0; i < current_node.get_feature_index().size(); i++){
			unknown_exist = false;
//			System.out.println(current_node.get_feature_index().get(i));
//			System.out.println(global_features.valueCount(current_node.get_feature_index().get(i)));
			for(int j = 0; j < current_node.get_label_index().size(); j++){
				if(global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(i)) == Matrix.MISSING){
					unknown_exist = true;
					extra_node = true;
				}
			}
			int[] total_instance = new int[global_features.valueCount(current_node.get_feature_index().get(i))];
			int[][] num_instances = new int[global_features.valueCount(current_node.get_feature_index().get(i))][global_labels.valueCount(0)];
			if(unknown_exist){
				total_instance = new int[global_features.valueCount(current_node.get_feature_index().get(i))+1];
				num_instances = new int[global_features.valueCount(current_node.get_feature_index().get(i))+1][global_labels.valueCount(0)];
			}
			for(int j = 0; j < current_node.total_instance(); j++){
				int whatever = -1;
				if(global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(i)) == Matrix.MISSING){
					whatever = global_features.valueCount(current_node.get_feature_index().get(i));
				}
				else{
					whatever = (int) global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(i));
				}
				total_instance[whatever] += 1;
				num_instances[whatever][(int) global_labels.get(current_node.get_label_index().get(j), 0)] += 1;
			}
			
			double infoa = 0;
			for(int j = 0; j < total_instance.length; j++){
				for(int k = 0; k < num_instances[j].length; k++){
					if(num_instances[j][k] != 0){
						infoa += ((double)total_instance[j]/current_node.get_label_index().size())*((double)num_instances[j][k]/total_instance[j]*Math.log((double)num_instances[j][k]/total_instance[j])/Math.log(2));
					}
				}
			}
//			System.out.println(infoa);
			infoa_list.add(-infoa);
//			for(int j = 0; j < total_instance.length; j++){
//				System.out.println(total_instance[j]);
//			}
//			for(int j = 0; j < num_instances.length; j++){
//				for(int k = 0; k < num_instances[j].length; k++){
//					System.out.println(num_instances[j][k]);
//				}
//			}
			//			System.out.println();			
		}
		
		// pick a feature with maximum gain
		int index = 0;
		double minimum = 100;
		for(int i = 0; i < infoa_list.size(); i++){
			if(infoa_list.get(i) < minimum){
				minimum = infoa_list.get(i);
				index = i;
			}
		}
		
//		current_node.print_label_index();
//		System.out.println("feature to split on: " + current_node.get_feature_index().get(index));
		current_node.set_feature_to_split(current_node.get_feature_index().get(index));
		
		ArrayList<Integer> feature_index = new ArrayList<>();
		for(int i = 0; i < current_node.get_feature_index().size(); i++){
			if(i != index){
				feature_index.add(current_node.get_feature_index().get(i));
			}
		}
		//		for(int i = 0; i < feature_left.size(); i++){
		//			System.out.println(feature_index.get(i));
		//		}

		// Generate children Nodes with corresponding instances in features
		int loop_count = global_features.valueCount(current_node.get_feature_index().get(index));
		if(extra_node){
			loop_count++;
		}
		for(int i = 0; i < loop_count; i++){
			ArrayList<Integer> label_index = new ArrayList<>();
			if(i == loop_count-1 && extra_node){
				for(int j = 0; j < current_node.get_label_index().size(); j++){
					if(global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(index)) == Matrix.MISSING){
						label_index.add(current_node.get_label_index().get(j));
					}
				}
			}
			else{
				for(int j = 0; j < current_node.get_label_index().size(); j++){
					if(global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(index)) == i){
						//					System.out.println(current_node.get_label_index().get(j));
						label_index.add(current_node.get_label_index().get(j));
						//					System.out.println(global_features.get(current_node.get_label_index().get(j), current_node.get_feature_index().get(index)));
					}
				}
			}
			Node new_node = new Node(feature_index,label_index);
			current_node.add_to_children(new_node);
		}
	}

	public void create_tree(Node current_node){
		if(current_node.get_info() == 0){
			// do nothing
		}
		else if(current_node.get_feature_index().size() == 0){
			// do nothing
		}
		else{
			calculate_infoa(current_node);
			for(int i = 0; i < current_node.get_children().size(); i++){
				create_tree(current_node.get_children().get(i));
			}
		}
	}
	
	public void traverse_tree(Node current_node){
		if(current_node.get_children().size() == 0){
			current_node.print_label_index();
		}
		else{
			System.out.println("feature selected: " + current_node.get_feature_to_split());
			current_node.print_label_index();
			
			for(int i = 0; i < current_node.get_children().size(); i++){
				traverse_tree(current_node.get_children().get(i));
			}
		}
	}
	
	public double calculate_accuracy(Matrix features, Matrix labels) throws Exception{
		int correct = 0;
		double[] prediction = new double[1];
		for(int i = 0; i < features.rows(); i++){
			double[] instance = features.row(i);
			int target = (int) labels.get(i, 0);
			predict(instance, prediction);
			int predict = (int) prediction[0];
			if(predict == target){
				correct++;
			}
		}
		return (double)correct/features.rows();
	}
	
	public void prune_recursively(Node current_node, Matrix features, Matrix labels) throws Exception{
		if(current_node.get_children().size() == 0){
			return;
		}
		else{
			for(int i = 0; i < current_node.get_children().size(); i++){
				prune_recursively(current_node.get_children().get(i), features, labels);
				pruning(current_node, features, labels);
			}
		}
	}
	
	public void pruning(Node current_node, Matrix features, Matrix labels) throws Exception{
		double current_accuracy = calculate_accuracy(features, labels);
		ArrayList<Node> temp = current_node.get_children();
		current_node.children = null;
		double new_accuracy = calculate_accuracy(features, labels);
		if(current_accuracy > new_accuracy){
			current_node.children = temp;
		}
		
	}
	
	public Matrix set_unknown_data(Matrix features){
		
		double[] unknown_array = new double[features.cols()];
		for(int i = 0; i < features.rows(); i++){
			for(int j = 0; j < features.cols(); j++){
				if(features.get(i, j)==1.0){
					unknown_array[j] += 1;
				}
			}
		}
		for(int i = 0; i < features.rows(); i++){
			for(int j = 0; j < features.cols(); j++){
				if(features.get(i, j)==Matrix.MISSING && (unknown_array[j]/features.rows()) > 0.5){
					features.set(i, j, 1.0);
//					System.out.println("majority yes");
				}
				else if(features.get(i, j)==Matrix.MISSING && (unknown_array[j]/features.rows()) < 0.5){
					features.set(i, j, 0.0);
//					System.out.println("majority no");
				}
			}
		}		
		
		return features;
		
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {

//		double train_percent = 0.9;
//		int train_size = (int)(train_percent*features.rows());
//		Matrix train_features = new Matrix(features, 0, 0, train_size, features.cols());
//		Matrix train_labels = new Matrix(labels, 0, 0, train_size, 1);
//		Matrix test_features = new Matrix(features, train_size, 0, features.rows()-train_size, features.cols());
//		Matrix test_labels = new Matrix(labels, train_size, 0, features.rows()-train_size, 1);
	
//		global_features = train_features;
//		global_labels = train_labels;
		features = set_unknown_data(features);
		
		global_features = features;
		global_labels = labels;
		
		
		// label_index contains the indexes of row
		ArrayList<Integer> label_index = new ArrayList<>();
		for(int i = 0; i < labels.rows(); i++){
			label_index.add(i);
		}
		// feature_index contains the available features
		ArrayList<Integer> feature_index = new ArrayList<>();
		for(int i = 0; i < features.cols(); i++){
			feature_index.add(i);
		}
		Node root = new Node(feature_index, label_index);
//		System.out.println("root info: " + root.get_info());		
//		System.out.println("num of instances: " + root.total_instance());
		create_tree(root);
//		traverse_tree(root);
		global_root = root;
	}
	
	public void predict_recursively(Node current_node, double[] features, double[] labels){
		if(current_node.children.size() == 0){
//			System.out.println(global_labels.get(current_node.get_label_index().get(1), 0));
//			System.out.println("feature left size: " + current_node.get_feature_index().size());
//			System.out.println("label left size: " + current_node.get_label_index().size());
//			System.out.println(current_node.get_output());
			labels[0] = current_node.get_output();
		}
		else{
			if(features[current_node.get_feature_to_split()] == Matrix.MISSING){
				if(global_features.valueCount(current_node.get_feature_to_split()) == current_node.get_children().size()){
//				System.out.println(current_node.get_children().get((int) features[current_node.get_feature_to_split()]));
					labels[0] = current_node.get_output();
				}
				else{
					predict_recursively(current_node.get_children().get(current_node.get_children().size()-1), features, labels);
				}
			}
			else{
//				System.out.println(current_node.get_children().get((int) features[current_node.get_feature_to_split()]));
				predict_recursively(current_node.get_children().get((int) features[current_node.get_feature_to_split()]), features, labels);
			}
		}
	}

	public void predict(double[] features, double[] labels) throws Exception {
		predict_recursively(global_root, features, labels);
	}

	@Override
	public double predict(double[] features, int target) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
}
