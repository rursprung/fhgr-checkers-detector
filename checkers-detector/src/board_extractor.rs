use BoardExtractorError::*;
use log::debug;
use opencv::{
    calib3d::find_homography_ext_def,
    core::{Point2f, Point2i, Size, ToInputArray, ToOutputArray, Vector},
    imgproc::warp_perspective_def,
    objdetect::ArucoDetector,
    prelude::*,
};
use std::{
    fmt::{Display, Formatter},
    iter::zip,
};

#[derive(Debug)]
pub enum BoardExtractorError {
    OpenCVError(opencv::Error),
}

impl From<opencv::Error> for BoardExtractorError {
    fn from(value: opencv::Error) -> Self {
        OpenCVError(value)
    }
}

impl Display for BoardExtractorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenCVError(_) => write!(f, "OpenCV internal error"),
        }
    }
}

impl std::error::Error for BoardExtractorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenCVError(err) => Some(err),
        }
    }
}

pub type Result<T> = std::result::Result<T, BoardExtractorError>;

#[derive(Debug, Clone)]
pub struct Config {
    /// The number of fields per row & column. Defines an n✕n board.
    pub num_fields_per_line: u8,
    /// Defines a fixed length for the edge of a field in the rectified image.
    pub px_per_field_edge: u8,
}

impl Config {
    fn num_columns_total(&self) -> u8 {
        // one for the aruco marker and one for the labels per side
        self.num_fields_per_line + 4
    }

    fn image_edge_length(&self) -> i32 {
        self.px_per_field_edge as i32 * self.num_columns_total() as i32
    }
}

fn get_outer_corners<M>(
    image: &M,
) -> Result<(
    Option<Vector<Point2i>>,
    (Vector<i32>, Vector<Vector<Point2f>>),
)>
where
    M: MatTrait + ToOutputArray + ToInputArray,
{
    let detector = ArucoDetector::new_def()?;
    let mut corners = Vector::<Vector<Point2f>>::default();
    let mut ids = Vector::<i32>::default();
    detector.detect_markers_def(image, &mut corners, &mut ids)?;

    if ids.len() != 4 {
        return Ok((None, (ids, corners)));
    }

    let mut result = Vector::<Point2i>::from_slice(&[
        Point2i::default(),
        Point2i::default(),
        Point2i::default(),
        Point2i::default(),
    ]);
    // TODO: validate that ids are in 0..=3 and that they are unique
    for (id, corner) in zip(ids.iter(), corners.iter()) {
        let corner = corner.get(id as usize)?;
        result.set(id as usize, Point2i::new(corner.x as i32, corner.y as i32))?;
    }

    Ok((Some(result), (ids, corners)))
}

pub fn extract_board<M>(image: &M, config: &Config) -> Result<Option<Mat>>
where
    M: MatTrait + ToOutputArray + ToInputArray,
{
    let (corners, ocv_markers) = get_outer_corners(image)?;

    if corners.is_none() {
        debug!(
            "failed to find (enough) corners! found the following corners: {:?}",
            ocv_markers.0
        );
        return Ok(None);
    }
    let corners = corners.unwrap();

    let l = config.image_edge_length();

    let destination_corners = Vector::<Point2i>::from_slice(&[
        Point2i::new(0, 0), // top left
        Point2i::new(l, 0), // top right
        Point2i::new(l, l), // bottom right
        Point2i::new(0, l), // bottom left
    ]);

    let homography = find_homography_ext_def(&corners, &destination_corners)?;

    let mut board_image = Mat::default();
    warp_perspective_def(image, &mut board_image, &homography, Size::new(l, l))?;

    Ok(Some(board_image))
}
