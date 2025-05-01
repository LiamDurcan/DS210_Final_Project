use std::error::Error;
fn euclidean_distance(vec_1: Vec<f64>, vec_2: Vec<f64>) -> Result<f64, Box<dyn Error>>{
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
/*
fn nearest_neighbors(data: Vec<Vec<f64>>, test_point:Vec<f64> num_neighbors:u32) -> Vec<u32> <- stores the indices of the nearest neighbors in the vector 
*/
mod tests{
    use super::*;
    #[test]
    fn test_euclidean_distance(){
        let vec_1 = vec![0.0,0.0,0.0,0.0];
        let vec_2 = vec![3.0,4.0,10.0,10.0];
        let distance = euclidean_distance(vec_1,vec_2).unwrap();
        assert_eq!(15.0, distance);//expected euclidean distance between these vectors is 15.0
    }
}
