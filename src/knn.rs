use std::error::Error;
use std::collections::HashMap; //needed to track votes for each label in knn function
fn euclidean_distance(vec_1: &Vec<f64>, vec_2: &Vec<f64>) -> Result<f64, Box<dyn Error>>{
    if vec_1.len() != vec_2.len(){
        return Err("One of the vectors is missing elements or has an extra element".to_string().into());
    }
    let mut sum:f64 = 0.0;
    for index in 0..vec_1.len(){
        sum+=(vec_1[index] - vec_2[index]).powf(2.0)//add distances between each components of the vectors
    }
    sum = sum.powf(0.5);
    Ok(sum)
}

fn nearest_vectors(train_data: &Vec<Vec<f64>>, test_point:&Vec<f64>, num_neighbors:u32) -> Vec<usize>{
    let mut distances: Vec<(usize,f64)> = Vec::new();
    for i in 0..train_data.len(){
        let distance = euclidean_distance(&train_data[i],&test_point).unwrap();
        distances.push((i,distance));
    }
    distances.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    let mut neighbor_indices:Vec<usize> = Vec::new();
    for i in 0..num_neighbors as usize{
        neighbor_indices.push(distances[i].0);
    }
    neighbor_indices
}

pub fn k_nearest_neighbors(train_data: &Vec<Vec<f64>>, test_data:&Vec<Vec<f64>>, train_labels:&Vec<bool>, test_labels:&Vec<bool>, num_neighbors:u32) -> (Vec<bool>, f32){
    let mut return_vec: Vec<bool> = Vec::new();
    let mut correct_predictions:f32 = 0.0;
    let mut total_predictions:f32 = test_labels.len() as f32;
    //go by index i instead of directly iterating through test_data because we need to match the data up with its label
    for i in 0..test_data.len(){
        let neighbor_indices = nearest_vectors(train_data,&test_data[i],num_neighbors);
        let mut neighbor_labels: Vec<bool> = Vec::new();
        for neighbor in neighbor_indices{
            neighbor_labels.push(train_labels[neighbor]);
        }
        let mut label_vote = HashMap::new();
        for label in neighbor_labels{
            *label_vote.entry(label).or_insert(0) +=1;
        }
        let mut sorted_label_vote:Vec<(bool,usize)> = label_vote.iter().map(|(key, val)| (*key, *val as usize)).collect();
        sorted_label_vote.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());//starting with b sorts in descending order
        let voted_label = sorted_label_vote[0].0;
        return_vec.push(voted_label);
        if voted_label == test_labels[i]{
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
        let vec_1 = vec![0.0,0.0,0.0,0.0];
        let vec_2 = vec![3.0,4.0,10.0,10.0];
        let distance = euclidean_distance(&vec_1,&vec_2).unwrap();
        assert_eq!(15.0, distance);//expected euclidean distance between these vectors is 15.0
    }

    #[test]
    fn test_nearest_vectors(){
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
