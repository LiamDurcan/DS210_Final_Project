use std::error::Error;
use std::collections::HashMap; //needed to track votes for each label in knn function

//Helper function for functions below
//finds the distance between two vectors
fn euclidean_distance(vec_1: &Vec<f64>, vec_2: &Vec<f64>) -> Result<f64, Box<dyn Error>>{
    if vec_1.len() != vec_2.len(){//Ensure they have the same dimensionality
        return Err("One of the vectors is missing elements or has an extra element".to_string().into());
    }
    let mut sum:f64 = 0.0;
    for index in 0..vec_1.len(){
        sum+=(vec_1[index] - vec_2[index]).powf(2.0)//add squared distances between each components of the vectors
    }
    sum = sum.powf(0.5);//square root this sum
    Ok(sum)
}

//Helper function for k_nearest_neighbors function
//Finds the nearest n (num_neighbors) vectors to the test_point vector. 
//Returns the indices of the closest vectors with respect to the entire train_data vector of vectors
fn nearest_vectors(train_data: &Vec<Vec<f64>>, test_point:&Vec<f64>, num_neighbors:u32) -> Vec<usize>{
    //Stores tuples with each vector in the test data's index and its distance to the test point.
    let mut distances: Vec<(usize,f64)> = Vec::new();
    for i in 0..train_data.len(){
        let distance = euclidean_distance(&train_data[i],&test_point).unwrap();
        distances.push((i,distance));
    }
    //sorts by distances (element 1) in ascending order (lowest distances at front)
    distances.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    //Adds all relevant indices to a vector that gets returned
    let mut neighbor_indices:Vec<usize> = Vec::new();
    for i in 0..num_neighbors as usize{
        neighbor_indices.push(distances[i].0);
    }
    neighbor_indices
}

/*Function that's called and tested in main.rs
Iterates through every point in test_data and feeds it through nearest_vectors

For each point...
Has the returned training vectors vote on the label for the test point
Compares the assigned/voted label to the test point's true label
Adds 1.0 to correct_predictions counter if labels are the same
Adds the prediction to a vector of predicted labels
Returns a tuple containing the vector of predicted labels (booleans) and the accuracy of the model
 */
pub fn k_nearest_neighbors(train_data: &Vec<Vec<f64>>, test_data:&Vec<Vec<f64>>, train_labels:&Vec<bool>, test_labels:&Vec<bool>, num_neighbors:u32) -> (Vec<bool>, f32){
    let mut return_vec: Vec<bool> = Vec::new();
    let mut correct_predictions:f32 = 0.0;//start at zero, increment by 1 every time
    let mut total_predictions:f32 = test_labels.len() as f32;//one prediction for each piece of test data
    //go by index i instead of directly iterating through test_data because we need to match the data up with its label
    for i in 0..test_data.len(){
        let neighbor_indices = nearest_vectors(train_data,&test_data[i],num_neighbors);
        let mut neighbor_labels: Vec<bool> = Vec::new();
        //get all the neighbors' labels in a vector
        for neighbor in neighbor_indices{
            neighbor_labels.push(train_labels[neighbor]);
        }
        //Track "Votes" or number of each training label in a HashMap (below)
        let mut label_vote = HashMap::new();
        //Increment each label by 1 
        for label in neighbor_labels{
            *label_vote.entry(label).or_insert(0) +=1;
        }
        //Convert the HashMap to a vector of tuples which can be sorted
        let mut sorted_label_vote:Vec<(bool,usize)> = label_vote.iter().map(|(key, val)| (*key, *val as usize)).collect();
        //Sort the vector in descending order (highest number of votes at start of vector)
        sorted_label_vote.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let voted_label = sorted_label_vote[0].0;//Label with most votes will now be at index 0 in the first tuple
        return_vec.push(voted_label);//Add predicted label to a vector to track predictions (in case they need reviewed)
        if voted_label == test_labels[i]{//Add 1.0 to correct predictions if correct
            correct_predictions += 1.0;
        }
    }

    let accuracy = correct_predictions/total_predictions;
    (return_vec,accuracy)
}

mod tests{
    use super::*;
    #[test]
    fn test_euclidean_distance(){
        //Defined two high-dimensionality vectors and calculated their distance by hand & on paper
        //assert_eq!() should pass
        let vec_1 = vec![0.0,0.0,0.0,0.0];
        let vec_2 = vec![3.0,4.0,10.0,10.0];
        let distance = euclidean_distance(&vec_1,&vec_2).unwrap();
        assert_eq!(15.0, distance);//expected euclidean distance between these vectors is 15.0
    }

    #[test]
    fn test_nearest_vectors(){
        //Had my test point be the origin for easy algebra.
        //Made first 3 points significantly closer
        //put all training vectors into a vector that would be the "training data"
        //Expected indices 0,1,2 in that order b/c vec_0 is closer than vec_1 etc. for test point
        let num_neighbors:u32 = 3;
        let test_point = vec![0.0,0.0,0.0];
        let vec_0:Vec<f64> = vec![1.0,0.0,0.0];
        let vec_1:Vec<f64> = vec![0.0,2.0,0.0];
        let vec_2:Vec<f64> = vec![0.0,0.0,3.0];
        let vec_3:Vec<f64> = vec![2.0,2.0,2.0];
        let vec_4:Vec<f64> = vec![5.0,4.0,12.0];
        let data_vec = vec![vec_0,vec_1,vec_2,vec_3,vec_4];
        let expected_output:Vec<usize> = vec![0,1,2];//nearest neighbors should be first 3 vectors in that order
        assert_eq!(expected_output,nearest_vectors(&data_vec,&test_point,3));
    }
    #[test]
    fn test_k_nearest_neighbors(){
        /*Had a test point at the origin with a true label and at 10,10,10 with a False label
        Wanted to ensure that the function would properly recognize correctly AND incorrectly labeled points
        Expected 1 of the two predictions (origin) to be correct

        Had all of the points closest to 1 be the same label, but only 1 point close to 10,10,10 be the
        same label, so with 3 or more neighbors 10 would be labeled incorrectly.

        To test final values, I created the tuple that I expected the function to return and used assert_eq()
        to compare this tuple to the function's true return value
        */
        let num_neighbors:u32 = 3;
        let test_vec_0 = vec![0.0,0.0,0.0];
        let test_vec_1= vec![10.0,10.0,10.0];
        let train_vec_0 = vec![1.0,1.0,1.0];
        let train_vec_1 = vec![2.0,2.0,2.0];
        let train_vec_2 = vec![5.0,5.0,5.0];
        let train_vec_3 = vec![8.0,8.0,8.0];
        let train_vec_4 = vec![6.0,6.0,6.0];

        let train_labels = vec![true,true,true,false,true];
        let test_labels = vec![true,false];
        let train_data = vec![train_vec_0,train_vec_1,train_vec_2,train_vec_3,train_vec_4];
        let test_data = vec![test_vec_0,test_vec_1];
        
        let expected_accuracy = 0.5;
        let expected_predicted_labels = vec![true,true];
        let expected_output = (expected_predicted_labels,expected_accuracy);

        assert_eq!(expected_output, k_nearest_neighbors(&train_data, &test_data, &train_labels, &test_labels, num_neighbors))
    }
}
