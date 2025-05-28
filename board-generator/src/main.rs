use Error::{InvalidConfig, OpenCVError};
use clap::Parser;
use opencv::{
    core::Size,
    core::{CV_8UC3, Point2i, Rect2i, Scalar},
    highgui,
    imgcodecs::imwrite_def,
    imgproc::{
        COLOR_GRAY2BGR, FONT_HERSHEY_SIMPLEX, INTER_LINEAR, LINE_AA, cvt_color_def, get_text_size,
        put_text, resize,
    },
    objdetect::PredefinedDictionaryType::DICT_4X4_50,
    prelude::*,
};
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    InvalidConfig,
}

impl From<opencv::Error> for Error {
    fn from(value: opencv::Error) -> Self {
        OpenCVError(value)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenCVError(_) => write!(f, "OpenCV internal error"),
            InvalidConfig => write!(f, "Invalid configuration specified"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenCVError(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

const COLOR_WHITE: Scalar = Scalar::new(255.0, 255.0, 255.0, 0.0);
const COLOR_RED: Scalar = Scalar::new(0.0, 0.0, 255.0, 0.0);
const COLOR_GREEN: Scalar = Scalar::new(0.0, 255.0, 0.0, 0.0);
const COLOR_ELECTRIC_BLUE: Scalar = Scalar::new(255.0, 255.0, 128.0, 0.0);
const COLOR_BLUE: Scalar = Scalar::new(255.0, 0.0, 0.0, 0.0);

fn draw_aruco_marker<M>(marker_id: i32, img_view: &mut M, config: &Config) -> Result<()>
where
    M: MatTrait + opencv::core::ToOutputArray + opencv::core::ToInputArray,
{
    let marker_edge_length_in_px = config.field_edge_length_in_px as i32;

    if img_view.size()?.width != marker_edge_length_in_px
        || img_view.size()?.height != marker_edge_length_in_px
    {
        panic!(
            "invalid image view, expected {}✕{} but got {}✕{}",
            marker_edge_length_in_px,
            marker_edge_length_in_px,
            img_view.size()?.width,
            img_view.size()?.height
        );
    }

    let dict = opencv::objdetect::get_predefined_dictionary(DICT_4X4_50)?;
    let mut marker = Mat::default();
    opencv::objdetect::generate_image_marker_def(
        &dict,
        marker_id,
        marker_edge_length_in_px,
        &mut marker,
    )?;
    cvt_color_def(&marker, img_view, COLOR_GRAY2BGR)?;
    Ok(())
}

fn draw_aruco_markers<M>(image: &mut M, config: &Config) -> Result<()>
where
    M: MatTrait + opencv::core::ToOutputArray + opencv::core::ToInputArray,
{
    // the numbering is done intentionally in this order:
    // this allows using it as the index to the aruco corners to always get the "outer" corner
    // of the marker, i.e. the one which identifies the outer corner of the actual chessboard + markers

    let marker_edge_length_in_px = config.field_edge_length_in_px as i32;
    let distance_from_image_edge_in_px = config.distance_from_image_edge_in_px as i32;

    // top left
    draw_aruco_marker(
        0,
        &mut image.roi_mut(Rect2i::new(
            distance_from_image_edge_in_px,
            distance_from_image_edge_in_px,
            marker_edge_length_in_px,
            marker_edge_length_in_px,
        ))?,
        config,
    )?;
    // top right
    draw_aruco_marker(
        1,
        &mut image.roi_mut(Rect2i::new(
            config.image_edge_length_in_px()
                - marker_edge_length_in_px
                - distance_from_image_edge_in_px,
            distance_from_image_edge_in_px,
            marker_edge_length_in_px,
            marker_edge_length_in_px,
        ))?,
        config,
    )?;
    // bottom right
    draw_aruco_marker(
        2,
        &mut image.roi_mut(Rect2i::new(
            config.image_edge_length_in_px()
                - marker_edge_length_in_px
                - distance_from_image_edge_in_px,
            config.image_edge_length_in_px()
                - marker_edge_length_in_px
                - distance_from_image_edge_in_px,
            marker_edge_length_in_px,
            marker_edge_length_in_px,
        ))?,
        config,
    )?;
    // bottom left
    draw_aruco_marker(
        3,
        &mut image.roi_mut(Rect2i::new(
            distance_from_image_edge_in_px,
            config.image_edge_length_in_px()
                - marker_edge_length_in_px
                - distance_from_image_edge_in_px,
            marker_edge_length_in_px,
            marker_edge_length_in_px,
        ))?,
        config,
    )?;

    Ok(())
}

fn draw_labels_vertically<M>(img: &mut M, config: &Config, column_center: i32) -> Result<()>
where
    M: MatTrait + opencv::core::ToInputOutputArray,
{
    let font_face = FONT_HERSHEY_SIMPLEX;
    let font_scale = 2.0;
    let thickness = 2;

    for row in 0..config.num_fields_per_line as i32 {
        let text = char::from(b'A' + row as u8).to_string();
        let text = text.as_str();
        let mut baseline = 0;
        let text_size = get_text_size(text, font_face, font_scale, thickness, &mut baseline)?;
        put_text(
            img,
            text,
            Point2i::new(
                column_center - text_size.width / 2,
                config.image_edge_length_in_px()
                    - 2 * config.field_edge_length_in_px as i32
                    - (row * config.field_edge_length_in_px as i32)
                    - config.field_edge_length_in_px as i32 / 2
                    + text_size.height / 2,
            ),
            font_face,
            font_scale,
            COLOR_RED,
            thickness,
            LINE_AA,
            false,
        )?;
    }

    Ok(())
}

fn draw_labels_horizontally<M>(img: &mut M, config: &Config, row_bottom: i32) -> Result<()>
where
    M: MatTrait + opencv::core::ToInputOutputArray,
{
    let font_face = FONT_HERSHEY_SIMPLEX;
    let font_scale = 2.0;
    let thickness = 2;

    for column in 0..config.num_fields_per_line as i32 {
        let text = char::from(b'1' + column as u8).to_string();
        let text = text.as_str();
        let mut baseline = 0;
        let text_size = get_text_size(text, font_face, font_scale, thickness, &mut baseline)?;
        put_text(
            img,
            text,
            Point2i::new(
                2 * config.field_edge_length_in_px as i32
                    + (column * config.field_edge_length_in_px as i32)
                    + config.field_edge_length_in_px as i32 / 2
                    - text_size.width / 2,
                row_bottom + text_size.height / 2,
            ),
            font_face,
            font_scale,
            COLOR_RED,
            thickness,
            LINE_AA,
            false,
        )?;
    }

    Ok(())
}

fn draw_label_background<M>(img: &mut M, config: &Config) -> Result<()>
where
    M: MatTrait + opencv::core::ToInputOutputArray,
{
    assert_eq!(img.size()?.height, img.size()?.width);
    let l = img.size()?.height;

    let label_background_color = &COLOR_ELECTRIC_BLUE;
    let field_edge_length_in_px = config.field_edge_length_in_px as i32;
    // left
    let mut roi = img.roi_mut(Rect2i::new(0, 0, field_edge_length_in_px, l))?;
    roi.set_to_def(label_background_color)?;
    // right
    let mut roi = img.roi_mut(Rect2i::new(
        l - field_edge_length_in_px,
        0,
        field_edge_length_in_px,
        l,
    ))?;
    roi.set_to_def(label_background_color)?;
    // top
    let mut roi = img.roi_mut(Rect2i::new(0, 0, l, field_edge_length_in_px))?;
    roi.set_to_def(label_background_color)?;
    // bottom
    let mut roi = img.roi_mut(Rect2i::new(
        0,
        l - field_edge_length_in_px,
        l,
        field_edge_length_in_px,
    ))?;
    roi.set_to_def(label_background_color)?;
    Ok(())
}

fn draw_labels<M>(img: &mut M, config: &Config) -> Result<()>
where
    M: MatTrait + opencv::core::ToInputOutputArray,
{
    // left
    draw_labels_vertically(
        img,
        config,
        config.distance_from_image_edge_in_px as i32
            + config.field_edge_length_in_px as i32
            + config.field_edge_length_in_px as i32 / 2,
    )?;

    // right
    draw_labels_vertically(
        img,
        config,
        config.image_edge_length_in_px()
            - config.distance_from_image_edge_in_px as i32
            - config.field_edge_length_in_px as i32
            - config.field_edge_length_in_px as i32 / 2,
    )?;

    // top
    draw_labels_horizontally(
        img,
        config,
        config.distance_from_image_edge_in_px as i32
            + config.field_edge_length_in_px as i32
            + config.field_edge_length_in_px as i32 / 2,
    )?;

    // bottom
    draw_labels_horizontally(
        img,
        config,
        config.image_edge_length_in_px()
            - config.distance_from_image_edge_in_px as i32
            - config.field_edge_length_in_px as i32
            - config.field_edge_length_in_px as i32 / 2,
    )?;

    Ok(())
}

/// Draw a chessboard on the provided image. The image is expected to have the right size.
fn draw_chessboard<M>(img_view: &mut M, config: &Config) -> Result<()>
where
    M: MatTrait + opencv::core::ToOutputArray,
{
    let field_edge_length_in_px = config.field_edge_length_in_px as i32;
    let num_fields = config.num_fields_per_line as i32;

    let mut field_type_a = true;

    if img_view.size()?.width != num_fields * field_edge_length_in_px
        || img_view.size()?.height != num_fields * field_edge_length_in_px
    {
        panic!(
            "invalid image view, expected {}✕{} but got {}✕{}",
            num_fields * field_edge_length_in_px,
            num_fields * field_edge_length_in_px,
            img_view.size()?.width,
            img_view.size()?.height
        );
    }

    for row in 0..num_fields {
        for col in 0..num_fields {
            let color = if field_type_a { COLOR_GREEN } else { COLOR_BLUE };

            let mut roi = img_view.roi_mut(Rect2i::new(
                row * field_edge_length_in_px,
                col * field_edge_length_in_px,
                field_edge_length_in_px,
                field_edge_length_in_px,
            ))?;
            roi.set_to_def(&color)?;

            // alternate color between fields
            field_type_a = !field_type_a;
        }
        // alternate color between rows (each line starts with a different color)
        field_type_a = !field_type_a;
    }

    Ok(())
}

pub fn generate_board(config: &Config) -> Result<Mat> {
    if config.field_edge_length_in_px < 10 || config.num_fields_per_line < 2 {
        return Err(InvalidConfig);
    }

    let mut image = Mat::new_rows_cols_with_default(
        config.image_edge_length_in_px(),
        config.image_edge_length_in_px(),
        CV_8UC3,
        COLOR_WHITE,
    )?;

    draw_chessboard(
        &mut image.roi_mut(Rect2i::new(
            config.distance_from_image_edge_in_px as i32
                + 2 * config.field_edge_length_in_px as i32,
            config.distance_from_image_edge_in_px as i32
                + 2 * config.field_edge_length_in_px as i32,
            config.chessboard_edge_length_in_px(),
            config.chessboard_edge_length_in_px(),
        ))?,
        config,
    )?;

    draw_label_background(
        &mut image.roi_mut(Rect2i::new(
            config.distance_from_image_edge_in_px as i32 + config.field_edge_length_in_px as i32,
            config.distance_from_image_edge_in_px as i32 + config.field_edge_length_in_px as i32,
            config.chessboard_edge_length_in_px() + 2 * config.field_edge_length_in_px as i32,
            config.chessboard_edge_length_in_px() + 2 * config.field_edge_length_in_px as i32,
        ))?,
        config,
    )?;

    draw_labels(&mut image, config)?;

    draw_aruco_markers(&mut image, config)?;

    Ok(image)
}

#[derive(Debug, Parser)]
pub struct Config {
    /// The number of fields per row & column. Creates an n✕n board.
    #[arg(short, long, default_value = "8")]
    num_fields_per_line: u8,
    #[arg(short = 'l', long, default_value = "100")]
    field_edge_length_in_px: u8,
    /// OpenCV cannot detect Aruco markers if they're at the edge of an image
    /// this isn't a problem when the image has been printed, and we see it through a camera,
    /// but it is a problem if we try to analyse the generated image directly.
    #[arg(short = 'e', long, default_value = "2")]
    distance_from_image_edge_in_px: u8,
    #[arg(short = 'f', long)]
    output_file: Option<std::path::PathBuf>,
}

impl Config {
    fn chessboard_edge_length_in_px(&self) -> i32 {
        self.num_fields_per_line as i32 * self.field_edge_length_in_px as i32
    }

    fn image_edge_length_in_px(&self) -> i32 {
        2 * self.distance_from_image_edge_in_px as i32
            + 2 * self.field_edge_length_in_px as i32 // aruco markers
            + 2 * self.field_edge_length_in_px as i32 // space between markers and board
            + self.chessboard_edge_length_in_px()
    }
}

fn main() -> Result<()> {
    let config = Config::parse();

    println!(
        "Generating {}✕{} board with {}px wide fields",
        config.num_fields_per_line, config.num_fields_per_line, config.field_edge_length_in_px
    );

    let image = generate_board(&config)?;

    let mut image_out = Mat::default();
    resize(
        &image,
        &mut image_out,
        Size::default(),
        0.5,
        0.5,
        INTER_LINEAR,
    )?;
    highgui::imshow("Board", &image_out)?;
    let key = highgui::wait_key_def()?;

    if key == 's' as i32 {
        if let Some(output_file) = config.output_file {
            imwrite_def(output_file.to_str().unwrap(), &image)?;
        }
    }

    Ok(())
}
