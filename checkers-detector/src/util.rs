use opencv::{
    Result,
    core::{Size, ToInputArray},
    highgui::imshow,
    imgproc::{INTER_LINEAR, resize},
    prelude::*,
};

pub fn resize_and_show<M>(window_name: &str, image: &M) -> Result<()>
where
    M: ToInputArray,
{
    let mut out = Mat::default();
    resize(image, &mut out, Size::default(), 0.5, 0.5, INTER_LINEAR)?;
    imshow(window_name, &out)?;
    Ok(())
}
