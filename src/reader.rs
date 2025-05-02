//Collaborators: None
/*This mod contains everything that's needed to read in the csv data.
Wtihin impl DataFrame, the read_csv function serves to populate a DataFrame
with all the values from the csv. The restrict_columns() function then allowed
me to filter down the DataFrame to the select columns I was trying to work with.
 */
use std::error::Error;
use std::fmt;
use std::process;
use csv::ReaderBuilder;
use std::collections::HashMap; //Using HashMaps to store all the values in a column of DataFrame

//Reused from HW8, this enum represents the different data types that could be read in from the csv
#[derive(Debug, Clone)]
pub enum ColumnVal {
    One(String),
    Two(bool),
    Three(f64),
    Four(i64),
}

//Decided to store values by column instead of row.
/*Reused from HW8, this struct stores all the csv data by column, making column operations
i.e. restrict_columns easier to carry out. It also stores the total number of rows (should
be equivalent to each column's length) and the order in which the columns are read from the csv.
 */
#[derive(Debug,Clone)]
pub struct DataFrame {
    pub num_rows:usize,
    pub columns: HashMap<String, Vec<ColumnVal>>,
    pub column_order: Vec<String> 
}
// For returning errors
#[derive(Debug)]
struct MyError(String);

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}
impl Error for MyError {}
/*reused some of the DataFrame code from HW8 to read in my CSV
This felt practical particularly because of the restrict_columns() function, which I included
This allowed me to narrow down my original dataframe into its numerical values and the column 
of labels I was intereested in predicting.
*/
impl DataFrame {
    pub fn new() -> Self {
        DataFrame{num_rows:0, columns: HashMap::new(), column_order:Vec::new(),}
    }

    //Returns Ok() if function succeeds or an error if it fails at any point
    pub fn read_csv(&mut self, path: &str, types: &Vec<u32>) -> Result<(), Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(false)
            .flexible(true)
            .from_path(path)?;
        let mut first_row = true;
        let mut column_titles: Vec<String> = vec![];
        for result in rdr.records() {
            // Notice that we need to provide a type hint for automatic
            // deserialization.
            let r = result.unwrap();
            let mut row: Vec<ColumnVal> = vec![];
            if first_row {
                for elem in r.iter() {
                    // These are the labels, all should be string values
                    column_titles.push(elem.to_string());
                    self.columns.insert(elem.to_string(), vec![]);
                    self.column_order.push(elem.to_string());
                }
                first_row = false;
                continue;
            }
            for (i, elem) in r.iter().enumerate() {
                match types[i] {
                    1 => row.push(ColumnVal::One(elem.to_string())),
                    2 => row.push(ColumnVal::Two(elem.parse::<bool>().unwrap())),
                    3 => row.push(ColumnVal::Three(elem.parse::<f64>().unwrap())),
                    4 => row.push(ColumnVal::Four(elem.parse::<i64>().unwrap())),
                    _ => return Err(Box::new(MyError("Unknown type".to_string()))),
                }
            }
            for (i, elem) in row.iter().enumerate(){
                self.columns.get_mut(&column_titles[i]).unwrap().push(elem.clone());
            }
            self.num_rows +=1;
            // Put the data into the dataframe
        }
        Ok(())
    }
    //included this function to create numerical_data_df and labels_df in main.rs once original DataFrame was created.
    pub fn restrict_columns(&mut self,column_names:Vec<String>) -> Result<(),Box <dyn Error>> {
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();
        for column in &column_names{
            //assigns column's vector to col variable by referencing and taking care of the Option via Some()
            if let Some(col_values) = self.columns.get(column){
                new_columns.insert(column.clone(), col_values.clone());
                new_column_order.push(column.clone());
                //clones the vector because ColumnVal doesn't have the Copy() trait
            }
            else{
                return Err(Box::new(MyError("One of the provided columns is not in the DataFrame".to_string())));
            }
        }
        self.columns = new_columns;
        self.column_order = column_names;
        Ok(())

    }
}