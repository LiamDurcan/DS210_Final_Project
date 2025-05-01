mod knn;
mod reader;
use reader::DataFrame;
fn main() {
    let types = vec![3,1,3,3,3,3,3,2,2,2];
    let mut df = DataFrame::new();
    df.read_csv("cleaned_version.csv",&types);

    let mut numerical_data_df = df.clone();
    let mut labels_df = df.clone();
    let mut data_headers = vec!["age","total_cholesterol","ldl","hdl","systolic_bp","diastolic_bp"];
    let mut label_header = vec!["heart_attack"];
    let label: Vec<String> = label_header.iter().map(|x| x.to_string()).collect();//convert the arguments to the appropriate type
    let data: Vec<String> = data_headers.iter().map(|x| x.to_string()).collect();//must be Vec<String> not Vec<&str>
    numerical_data_df.restrict_columns(data);
    labels_df.restrict_columns(label);

}
