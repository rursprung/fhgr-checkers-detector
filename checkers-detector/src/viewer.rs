use Error::*;
use crate::board_extractor::{extract_board, BoardExtractorError};
use crate::camera_control::Esp32Cam;
use crate::detector::{BoardLayout, CalibratedDetector, FieldOccupancy, FieldPosition};
use crate::util::{field_mask_roi, resize_and_show};
use crate::{detector, CameraType, PX_PER_FIELD_EDGE};
use lazy_static::lazy_static;
use log::{debug, info, warn};
use opencv::highgui::wait_key_def;
use opencv::imgcodecs::imread_def;
use opencv::{
    core::{Point2i, Rect2i, Scalar, ToInputOutputArray, mean_std_dev_def},
    highgui::{MouseEventTypes, named_window_def, set_mouse_callback, wait_key},
    imgproc::{COLOR_RGB2HSV, FONT_HERSHEY_COMPLEX, LINE_8, cvt_color_def, line, put_text_def},
    prelude::*,
    videoio::VideoCapture,
};
use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::sync::Mutex;
use url::Url;

lazy_static! {
    static ref VIEWER: Mutex<BoardViewer> = Mutex::new(BoardViewer::new().unwrap());
}

const COLOR_GREEN: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0);

#[derive(Debug)]
pub enum Error {
    UrlParseError(url::ParseError),
    UrlMustBeBaseUrl(String, String),
    ImageAcquisitionFailure(Option<opencv::Error>),
    OtherOpenCVError(opencv::Error),
    BoardNotFound,
    InternalDetectionError(detector::DetectorError),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            UrlParseError(_) => write!(f, "Url parse error"),
            UrlMustBeBaseUrl(url, expected_base_url) => write!(
                f,
                "Camera URLs do not match! expected {} but got {}",
                expected_base_url, url
            ),
            ImageAcquisitionFailure(_) => write!(f, "image acquisition failed"),
            OtherOpenCVError(_) => write!(f, "Generic OpenCV Error"),
            BoardNotFound => write!(
                f,
                "Board not found in image. Are all aruco markers in view?"
            ),
            InternalDetectionError(_) => write!(f, "internal detection error"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UrlParseError(e) => Some(e),
            ImageAcquisitionFailure(Some(e)) => Some(e),
            OtherOpenCVError(e) => Some(e),
            InternalDetectionError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<url::ParseError> for Error {
    fn from(e: url::ParseError) -> Self {
        UrlParseError(e)
    }
}

impl From<BoardExtractorError> for Error {
    fn from(e: BoardExtractorError) -> Self {
        use crate::board_extractor::BoardExtractorError::*;
        match e {
            OpenCVError(e) => OtherOpenCVError(e),
        }
    }
}

impl From<detector::DetectorError> for Error {
    fn from(e: detector::DetectorError) -> Self {
        use crate::detector::DetectorError::*;
        match e {
            OpenCVError(e) => OtherOpenCVError(e),
            e => InternalDetectionError(e),
        }
    }
}

/// Default to [`OtherOpenCVError`] unless `map_err` is used explicitly.
impl From<opencv::Error> for Error {
    fn from(e: opencv::Error) -> Self {
        OtherOpenCVError(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Config {
    num_fields_per_line: u8,
    px_per_field_edge: u8,
}

impl Config {
    fn num_columns_total(&self) -> u8 {
        // one for the aruco marker and one for the labels per side
        self.num_fields_per_line + 4
    }

    fn image_edge_length(&self) -> i32 {
        PX_PER_FIELD_EDGE as i32 * self.num_columns_total() as i32
    }
}

#[derive(Debug, Clone)]
pub struct BoardViewer {
    config: Option<Config>,
    detector: Option<CalibratedDetector>,
    original_image: Option<Mat>,
    board: Option<Mat>,
    result: Option<BoardLayout>,
}

impl BoardViewer {
    fn new() -> Result<Self> {
        named_window_def("board")?;
        set_mouse_callback("board", Some(Box::new(Self::handle_mouse_cb)))?;
        Ok(Self {
            config: None,
            detector: None,
            original_image: None,
            board: None,
            result: None,
        })
    }

    pub fn init(
        &mut self,
        detector: CalibratedDetector,
        num_fields_per_line: u8,
        px_per_field_edge: u8,
    ) {
        if self.detector.is_some() {
            panic!("Viewer is already initialized!");
        }

        self.detector = Some(detector);
        self.config = Some(Config {
            num_fields_per_line,
            px_per_field_edge,
        });
    }

    fn handle_mouse_cb(event: i32, x: i32, y: i32, flags: i32) {
        VIEWER
            .lock()
            .unwrap()
            .handle_mouse(event, x, y, flags);
    }

    fn handle_mouse(&self, event: i32, x: i32, y: i32, _flags: i32) {
        let pos = FieldPosition::try_from_px(
            2 * x as u32,
            2 * y as u32,
            self.config.unwrap().px_per_field_edge,
            self.config.unwrap().num_fields_per_line,
        )
        .unwrap();
        if event == MouseEventTypes::EVENT_LBUTTONDOWN as i32 {
            if let Some(result) = &self.result {
                if pos.is_on_board() {
                    info!("Selected field {}: {}", pos, result.get(&pos).unwrap());
                }
            }
        }
        #[cfg(feature = "show_debug_screens")]
        if event == MouseEventTypes::EVENT_RBUTTONDOWN as i32 {
            let (mean, std_dev) = self.get_mean_hsv_at_pos(&pos).unwrap();
            info!(
                "Mean HSV at field {}: mean = {:?}, std_dev = {:?}",
                pos, mean, std_dev
            );
        }
        #[cfg(feature = "show_debug_screens")]
        if event == MouseEventTypes::EVENT_MBUTTONDOWN as i32 {
            let (mean, std_dev) = self.get_mean_hsv_at_coord(x, y).unwrap();
            info!(
                "Mean HSV at [{},{}] (in field {}): mean = {:?}, std_dev = {:?}",
                x, y, pos, mean, std_dev
            );
        }
    }

    fn get_mean_std_dev_hsv_in_roi(&self, roi: Rect2i) -> Result<(Scalar, Scalar)> {
        let mut board_hsv = Mat::default();
        cvt_color_def(self.board.as_ref().unwrap(), &mut board_hsv, COLOR_RGB2HSV)?;

        let mut mean = Scalar::default();
        let mut std_dev = Scalar::default();
        mean_std_dev_def(&board_hsv.roi(roi)?, &mut mean, &mut std_dev)?;
        Ok((mean, std_dev))
    }

    #[allow(unused)]
    fn get_mean_hsv_at_coord(&self, x: i32, y: i32) -> Result<(Scalar, Scalar)> {
        let board_width = self.board.as_ref().unwrap().rows();
        let roi_half_width_px = 20;
        let roi = Rect2i::new(
            max(x - roi_half_width_px, 0),
            max(y - roi_half_width_px, 0),
            min(x + roi_half_width_px, board_width),
            min(y + roi_half_width_px, board_width),
        );
        self.get_mean_std_dev_hsv_in_roi(roi)
    }

    #[allow(unused)]
    fn get_mean_hsv_at_pos(&self, pos: &FieldPosition) -> Result<(Scalar, Scalar)> {
        let roi = field_mask_roi(pos, self.config.unwrap().px_per_field_edge);
        self.get_mean_std_dev_hsv_in_roi(roi)
    }

    /// Handle an individual frame.
    fn handle_frame(&mut self, frame: Mat) -> Result<()> {
        let board = extract_board(
            &frame,
            &crate::board_extractor::Config {
                num_fields_per_line: self.config.unwrap().num_fields_per_line,
                px_per_field_edge: self.config.unwrap().px_per_field_edge,
            },
        )?;
        if board.is_none() {
            warn!("failed to find board! skipping frame");
            return Err(BoardNotFound);
        }
        let board = board.unwrap();
        self.original_image = Some(frame);

        let result = self.detector.as_ref().unwrap().detect_pieces(&board)?;

        let mut board_annotated = Mat::default();
        board.copy_to(&mut board_annotated)?;
        self.annotate_image(&mut board_annotated, &result)?;

        resize_and_show("board", &board_annotated)?;

        debug!("{}", result);

        self.board = Some(board);
        self.result = Some(result);

        Ok(())
    }

    fn annotate_image<M>(&self, image: &mut M, layout: &BoardLayout) -> Result<()>
    where
        M: MatTrait + ToInputOutputArray,
    {
        self.overlay_grid(image)?;

        for (pos, field) in layout.field_iter() {
            if let FieldOccupancy::Occupied(_, _) = field {
                let text = format!("{}", field);
                let pos = Point2i::new(
                    PX_PER_FIELD_EDGE as i32 * pos.col_in_img() as i32 + 10,
                    PX_PER_FIELD_EDGE as i32 * (pos.row_in_img() as i32 + 1) - 10,
                );
                put_text_def(
                    image,
                    text.as_str(),
                    pos,
                    FONT_HERSHEY_COMPLEX,
                    3.0,
                    COLOR_GREEN,
                )?;
            }
        }

        Ok(())
    }

    fn overlay_grid<M>(&self, image: &mut M) -> Result<()>
    where
        M: MatTrait + ToInputOutputArray,
    {
        let config = self.config.as_ref().unwrap();
        for i in 1..config.num_columns_total() as i32 {
            // horizontal line
            line(
                image,
                Point2i::new(0, i * config.px_per_field_edge as i32),
                Point2i::new(
                    config.image_edge_length(),
                    i * config.px_per_field_edge as i32,
                ),
                COLOR_GREEN,
                3,
                LINE_8,
                0,
            )?;
            // vertical line
            line(
                image,
                Point2i::new(i * config.px_per_field_edge as i32, 0),
                Point2i::new(
                    i * config.px_per_field_edge as i32,
                    config.image_edge_length(),
                ),
                COLOR_GREEN,
                3,
                LINE_8,
                0,
            )?;
        }

        Ok(())
    }
}

/// Tries to open the specified video input and stream it while it lasts.
pub fn handle_video_input(video_input: &str, camera_type: Option<CameraType>) -> Result<()> {
    let video_input = match camera_type {
        Some(CameraType::Esp32Cam) => {
            let url = Url::parse(video_input).unwrap();
            let base_url = Url::parse(format!("http://{}", url.host().unwrap()).as_str()).unwrap();
            if url != base_url {
                return Err(UrlMustBeBaseUrl(url.to_string(), base_url.to_string()));
            }

            Esp32Cam::init(&base_url);

            let stream_url = format!("http://{}:81/stream", base_url.host().unwrap());
            stream_url
        }
        _ => video_input.to_string(),
    };

    let mut capture = match video_input.parse::<i32>() {
        Ok(i) => {
            debug!("opening ID based video capture {}", i);
            VideoCapture::new_def(i)
        }
        Err(_) => {
            debug!("opening path or URL based video capture {}", video_input);
            VideoCapture::from_file_def(&video_input)
        }
    }
    .map_err(|e| ImageAcquisitionFailure(Some(e)))?;
    if !capture
        .is_opened()
        .map_err(|e| ImageAcquisitionFailure(Some(e)))?
    {
        return Err(ImageAcquisitionFailure(None));
    }

    loop {
        let mut image = Mat::default();
        capture
            .read(&mut image)
            .map_err(|e| ImageAcquisitionFailure(Some(e)))?;

        resize_and_show("original image", &image)?;

        match VIEWER.lock().unwrap().handle_frame(image) {
            Ok(_) | Err(BoardNotFound) => {}
            Err(e) => return Err(e),
        }

        let key = wait_key(1)?;
        if key == 'q' as i32 {
            return Ok(());
        }
    }
}

pub fn init_viewer(detector: CalibratedDetector, num_fields_per_line: u8, px_per_field_edge: u8) {
    VIEWER
        .lock()
        .unwrap()
        .init(detector, num_fields_per_line, px_per_field_edge);
}

pub fn handle_single_image(image_path: &str) -> Result<()> {
    let image = imread_def(image_path).map_err(|e| ImageAcquisitionFailure(Some(e)))?;
    VIEWER.lock().unwrap().handle_frame(image)?;
    wait_key_def()?;
    Ok(())
}
