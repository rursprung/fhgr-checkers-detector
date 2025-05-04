use opencv::{
    core::{Scalar, CV_8UC3},
    highgui,
    prelude::*,
    Result,
};

fn main() -> Result<()> {
    let image = Mat::new_rows_cols_with_default(
        100,
        100,
        CV_8UC3,
        Scalar::from((255, 255, 255)),
    )?;

    highgui::imshow("hello opencv!", &image)?;
    highgui::wait_key_def()?;
    Ok(())
}
