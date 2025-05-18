use crate::detector::FieldPosition;
use opencv::core::Rect2i;
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

pub fn field_mask_roi(pos: &FieldPosition, px_per_field_edge: u8) -> Rect2i {
    Rect2i::new(
        px_per_field_edge as i32 * pos.col_in_img() as i32,
        px_per_field_edge as i32 * (pos.row_in_img() as i32 - 1),
        px_per_field_edge as _,
        2 * px_per_field_edge as i32,
    )
}
