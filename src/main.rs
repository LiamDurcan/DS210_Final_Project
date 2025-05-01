mod knn;
mod reader;
use reader::DataFrame;
use crate::reader::ColumnVal;
//use rand::Rng; originally used this to split the data
use rand::seq::SliceRandom;
fn main() {
    //Define types vector based on each column in the csv (in order)
    let types = vec![3,1,3,3,3,3,3,2,2,2];
    let mut df = DataFrame::new();
    df.read_csv("cleaned_version.csv",&types); //read in csv and update DataFrame

    //clones df twice and filters the clones to a numerical df and a df containing each row's label
    let mut numerical_data_df = df.clone();
    let mut labels_df = df.clone();
    let mut data_headers = vec!["age","total_cholesterol","ldl","hdl","systolic_bp","diastolic_bp"];
    let mut label_header = vec!["heart_attack"];
    let label: Vec<String> = label_header.iter().map(|x| x.to_string()).collect();//convert the arguments to the appropriate type
    let data: Vec<String> = data_headers.iter().map(|x| x.to_string()).collect();//must be Vec<String> not Vec<&str>
    numerical_data_df.restrict_columns(data);
    labels_df.restrict_columns(label);

    //Turn the numerical_data_df into a Vector of vectors from its original struct form
    let mut numerical_rows_vec:Vec<Vec<f64>> = vec![vec![]; numerical_data_df.num_rows];
    for column in numerical_data_df.column_order{
        let column_data = numerical_data_df.columns.get(&column).unwrap();
        for i in 0..numerical_data_df.num_rows{
            let num = match &column_data[i] {
                ColumnVal::Three(f) => *f,
                ColumnVal::Four(n) => *n as f64, // converts integers to float. Shouldn't be necessary but here just in case
                _ => panic!("Expected numeric column, but found non-numeric data."),
            };
            numerical_rows_vec[i].push(num);
        }
    }

    //Convert the labels to Vec<bool> from the labels_df dataframe.
    let mut labels: Vec<bool> = Vec::new();
    //for loop uses an iterator so we can use i as the index
    for i in 0..labels_df.num_rows{
        //gets the vector of ColumnVals from the hashmap (first and only column so use index 0), iterate over each column in vector
        match &labels_df.columns.get(&labels_df.column_order[0]).expect("Column not found.")[i]{
            ColumnVal::Two(value) => labels.push(*value),
            _ => panic!("Expected boolean label"), //Column should be entirely booleans
        }
    }

    //Split the data so that 20% is test data, 80% is training data
    /*
    let mut rng = rand::thread_rng();
    let random_num = rng.gen_range(1..=5); //Randomly select 1 in every 5 rows
    let mut numerical_test_rows:Vec<Vec<f64>> = Vec::new();
    let mut numerical_train_rows:Vec<Vec<f64>> = Vec::new();
    let mut test_labels: Vec<bool> = Vec::new();
    let mut train_labels:Vec<bool> = Vec::new();
    for i in 0..numerical_rows_vec.len(){
        if i%5 == random_num{//20% of data will become test data, the rest is training data
            numerical_test_rows.push(numerical_rows_vec[i].clone());
            test_labels.push(labels[i]);
        }
        else{
            numerical_train_rows.push(numerical_rows_vec[i].clone());
            train_labels.push(labels[i]);
        }
    }*/
    //randomly splits data so 20% is used to test. 
    //Had Help form ChatGPT using shuffle(). See Write-Up for details and my original method of splitting data
    //Created a vector of tuples to pair each numerical row with its label from the two dfs when splitting
    let mut rng = rand::thread_rng();//needed for shuffle function.
    let mut data_label_pairs: Vec<(Vec<f64>, bool)> = numerical_rows_vec.into_iter().zip(labels.into_iter()).collect();
    data_label_pairs.shuffle(&mut rng); // shuffle the paired data
    
    let test_size = (data_label_pairs.len() as f64 * 0.2).round() as usize; //gets 20% of size of dataset
    let (test_data, train_data) = data_label_pairs.split_at(test_size);
    
    // Unzip array of tuples back into two arrays of features and labels for both test and training datasets 
    let (numerical_test_rows, test_labels): (Vec<_>, Vec<_>) = test_data.iter().cloned().unzip();
    let (numerical_train_rows, train_labels): (Vec<_>, Vec<_>) = train_data.iter().cloned().unzip();

    //split the execution up over two lines to improve readability
    let (predictions, accuracy) = knn::k_nearest_neighbors
    (&numerical_train_rows,&numerical_test_rows,&train_labels,&test_labels,3);
    println!("KNN Accuracy on Randomized, Synthetic Data: {}", accuracy); //report Overall knn accuracy

}
