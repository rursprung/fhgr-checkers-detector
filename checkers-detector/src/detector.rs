use DetectorError::*;
use array2d::Array2D;
use log::debug;
use opencv::{
    core::{Rect2i, Scalar, Size2i, ToInputArray, bitwise_and_def, count_non_zero, in_range},
    imgproc::{
        COLOR_RGB2HSV, MORPH_OPEN, MORPH_RECT, cvt_color_def, get_structuring_element_def,
        morphology_ex_def,
    },
    prelude::*,
};
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum DetectorError {
    OpenCVError(opencv::Error),
    IndexOutOfBounds(usize, usize, usize),
    InvalidPosition,
}

impl From<opencv::Error> for DetectorError {
    fn from(value: opencv::Error) -> Self {
        OpenCVError(value)
    }
}

impl Display for DetectorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenCVError(_) => write!(f, "OpenCV internal error"),
            IndexOutOfBounds(actual, min, max) => write!(
                f,
                "index out of bounds! Was {actual} but must be between {min} and {max}"
            ),
            InvalidPosition => write!(f, "Invalid position"),
        }
    }
}

impl std::error::Error for DetectorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenCVError(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, DetectorError>;

#[allow(unused)] // depends on feature flags
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum DebugFieldConfig {
    None,
    All,
    Specific(FieldPosition),
}

impl DebugFieldConfig {
    pub fn matches(&self, other: &FieldPosition) -> bool {
        match self {
            DebugFieldConfig::None => false,
            DebugFieldConfig::All => true,
            DebugFieldConfig::Specific(pos) => pos == other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Config {
    /// The number of fields per row & column. Defines an n✕n board.
    pub num_fields_per_line: u8,
    /// Defines a fixed length for the edge of a field in the rectified image.
    pub px_per_field_edge: u8,
    /// Whether specific fields should be debugged or not.
    pub debug_field_config: DebugFieldConfig,
}

impl Config {
    fn num_columns_total(&self) -> u8 {
        // one for the aruco marker and one for the labels per side
        self.num_fields_per_line + 4
    }

    fn image_edge_length(&self) -> i32 {
        self.px_per_field_edge as i32 * self.num_columns_total() as i32
    }

    fn px_per_field(&self) -> i32 {
        self.px_per_field_edge as i32 * self.px_per_field_edge as i32
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum PieceColour {
    Black,
    White,
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum PieceType {
    Man,
    #[allow(unused)]
    King,
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum FieldOccupancy {
    WrongType,
    Empty,
    Occupied(PieceColour, PieceType),
}

impl Display for FieldOccupancy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldOccupancy::WrongType => write!(f, " "),
            FieldOccupancy::Empty => write!(f, " "),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::Man) => write!(f, "b"),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::King) => write!(f, "B"),
            FieldOccupancy::Occupied(PieceColour::White, PieceType::Man) => write!(f, "w"),
            FieldOccupancy::Occupied(PieceColour::White, PieceType::King) => write!(f, "W"),
        }
    }
}

/// Represents a position on the board.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct FieldPosition {
    /// row, counted from the top (opposite direction of what is printed on the board!).
    row: u8,
    /// column, counted from the left.
    col: u8,
    /// number of fields per line (and row) - needed to correctly label the row.
    num_fields_per_line: u8,
}

impl FieldPosition {
    /// Create based on position on the board, indexed with (0,0) in the top left field.
    #[allow(unused)]
    pub fn try_new(row: u8, col: u8, num_fields_per_line: u8) -> Result<Self> {
        if row > num_fields_per_line - 1 {
            return Err(IndexOutOfBounds(
                row as usize,
                0,
                num_fields_per_line as usize - 1,
            ));
        }
        if col > num_fields_per_line - 1 {
            return Err(IndexOutOfBounds(
                col as usize,
                0,
                num_fields_per_line as usize - 1,
            ));
        }
        Ok(Self {
            row,
            col,
            num_fields_per_line,
        })
    }

    /// Create based on position on the board, indexed with (2,2) in the top left field.
    #[allow(unused)]
    pub fn try_from_img_pos(img_row: u8, img_col: u8, num_fields_per_line: u8) -> Result<Self> {
        if img_row < 2 || img_row > num_fields_per_line + 2 - 1 {
            return Err(IndexOutOfBounds(
                img_row as usize,
                2,
                num_fields_per_line as usize + 2 - 1,
            ));
        }
        if img_col < 2 || img_col > num_fields_per_line + 2 - 1 {
            return Err(IndexOutOfBounds(
                img_col as usize,
                2,
                num_fields_per_line as usize + 2 - 1,
            ));
        }
        Ok(Self {
            row: img_row - 2,
            col: img_col - 2,
            num_fields_per_line,
        })
    }

    #[allow(unused)]
    pub fn try_from_str(pos: &str, num_fields_per_line: u8) -> Result<Self> {
        if pos.len() != 2 {
            return Err(InvalidPosition);
        }

        let row = &pos.as_bytes()[0] - (b'A' + num_fields_per_line - 1);
        let col = pos[1..2].parse::<u8>().map_err(|_| InvalidPosition)? - 1;

        Self::try_new(row, col, num_fields_per_line)
    }

    pub fn row(&self) -> u8 {
        self.row
    }

    pub fn col(&self) -> u8 {
        self.col
    }

    pub fn row_in_img(&self) -> u8 {
        self.row + 2
    }

    pub fn col_in_img(&self) -> u8 {
        self.col + 2
    }

    pub fn is_black_field(&self) -> bool {
        if self.row % 2 == 0 {
            self.col % 2 == 1
        } else {
            self.col % 2 == 0
        }
    }
}

impl Display for FieldPosition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let row = char::from(b'A' + self.num_fields_per_line - 1 - self.row);
        let col = char::from(b'1' + self.col);
        write!(f, "{}{}", row, col)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoardLayout(Array2D<FieldOccupancy>);

impl BoardLayout {
    fn new(num_fields_per_line: u8) -> Self {
        Self(Array2D::filled_with(
            FieldOccupancy::WrongType,
            num_fields_per_line as _,
            num_fields_per_line as _,
        ))
    }

    #[allow(unused)]
    pub fn field_iter(&self) -> impl Iterator<Item = (FieldPosition, &FieldOccupancy)> {
        self.field_pos_iter()
            .map(|pos| (pos, self.get(&pos).unwrap()))
    }

    pub fn field_pos_iter(&self) -> impl Iterator<Item = FieldPosition> {
        let num_fields_per_line = self.0.row_len() as u8;

        (0..num_fields_per_line)
            .into_iter()
            .map(move |row| {
                (0..num_fields_per_line).into_iter().map(move |col| {
                    FieldPosition::try_new(
                        row.clone() as u8,
                        col.clone() as u8,
                        num_fields_per_line.clone(),
                    )
                    .unwrap()
                })
            })
            .flatten()
    }

    fn get(&self, pos: &FieldPosition) -> Option<&FieldOccupancy> {
        self.0.get(pos.row() as usize, pos.col() as usize)
    }

    fn set(&mut self, pos: &FieldPosition, field_occupancy: FieldOccupancy) {
        self.0
            .set(pos.row() as usize, pos.col() as usize, field_occupancy)
            .unwrap();
    }

    fn print_column_header(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "   ")?;
        for column in 0..self.0.row_len() {
            write!(f, " {}  ", char::from(b'1' + column as u8).to_string())?;
        }
        writeln!(f)?;
        Ok(())
    }

    fn print_row(&self, f: &mut Formatter<'_>, row: usize) -> std::fmt::Result {
        let row_label = char::from(b'A' + (self.0.column_len() - row - 1) as u8);
        write!(f, "{} |", row_label)?;
        for column in 0..self.0.row_len() {
            write!(f, " {} |", self.0.get(row, column).unwrap())?;
        }
        write!(f, " {} ", row_label)?;
        writeln!(f)?;
        Ok(())
    }

    fn print_row_separator(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  |{}", "---|".repeat(self.0.row_len()))
    }
}

impl Display for BoardLayout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        self.print_column_header(f)?;
        self.print_row_separator(f)?;
        for row in 0..self.0.column_len() {
            self.print_row(f, row)?;
            self.print_row_separator(f)?;
        }
        self.print_column_header(f)?;
        Ok(())
    }
}

#[derive(Debug)]
struct InitialImageProcessor {
    config: Config,
    board_hsv: Mat,
}

impl InitialImageProcessor {
    fn new<M>(image: &M, config: Config) -> Result<Self>
    where
        M: MatTrait + ToInputArray,
    {
        let mut board_hsv = Mat::default();
        cvt_color_def(image, &mut board_hsv, COLOR_RGB2HSV)?;
        Ok(Self { config, board_hsv })
    }

    fn get_mask_for_board_area(&self, include_labels: bool) -> Result<Mat> {
        let mut mask = Mat::new_size_with_default(
            self.board_hsv.size()?,
            u8::opencv_type(),
            Scalar::all(0.0),
        )?;

        let offset = self.config.px_per_field_edge as i32 * if include_labels { 1 } else { 2 };

        let mut roi = mask.roi_mut(Rect2i::new(
            offset,
            offset,
            self.config.image_edge_length() - 2 * offset,
            self.config.image_edge_length() - 2 * offset,
        ))?;
        roi.set_to_def(&Scalar::all(u8::MAX as f64))?;

        Ok(mask)
    }

    fn get_mask_for_colour(
        &self,
        lower_bound_hsv: &Scalar,
        upper_bound_hsv: &Scalar,
    ) -> Result<Mat> {
        let mut mask = Mat::default();
        in_range(&self.board_hsv, lower_bound_hsv, upper_bound_hsv, &mut mask)?;

        let mut mask_morph = Mat::default();
        morphology_ex_def(
            &mask,
            &mut mask_morph,
            MORPH_OPEN,
            &get_structuring_element_def(MORPH_RECT, Size2i::new(10, 10))?,
        )?;

        Ok(mask_morph)
    }

    fn get_mask_for_pieces(
        &self,
        lower_bound_hsv: &Scalar,
        upper_bound_hsv: &Scalar,
    ) -> Result<Mat> {
        let mut result = Mat::default();
        bitwise_and_def(
            &self.get_mask_for_colour(lower_bound_hsv, upper_bound_hsv)?,
            &self.get_mask_for_board_area(true)?,
            &mut result,
        )?;

        Ok(result)
    }

    fn get_mask_for_black_pieces(&self) -> Result<Mat> {
        self.get_mask_for_pieces(&Scalar::all(0.0), &Scalar::new(180.0, 255.0, 50.0, 0.0))
    }

    fn process(self) -> Result<BoardProcessor> {
        Ok(BoardProcessor {
            config: self.config,
            mask_blacks: self.get_mask_for_black_pieces()?,
            _mask_whites: Default::default(), // TODO
        })
    }
}

struct BoardProcessor {
    config: Config,
    mask_blacks: Mat,
    _mask_whites: Mat,
}

impl BoardProcessor {
    fn detect_pieces(self) -> Result<BoardLayout> {
        #[cfg(feature = "show_debug_screens")]
        {
            use opencv::{
                core::Size,
                highgui::imshow,
                imgproc::{INTER_LINEAR, resize},
            };

            let mut out = Mat::default();
            resize(
                &self.mask_blacks,
                &mut out,
                Size::default(),
                0.5,
                0.5,
                INTER_LINEAR,
            )?;
            imshow("mask blacks", &out)?;
        }

        let mut result = BoardLayout::new(self.config.num_fields_per_line);
        // save the positions due to mutability issues (we cannot just take the iter and map it)
        let field_positions = result
            .field_pos_iter()
            .filter(|pos| pos.is_black_field())
            .collect::<Vec<_>>();

        for pos in field_positions.iter() {
            let field_occupancy = self.detect_pieces_on_field(&pos)?;
            debug!("field occupancy at {} is {}", pos, field_occupancy,);
            result.set(&pos, field_occupancy);
        }

        Ok(result)
    }

    fn field_mask_roi(&self, pos: &FieldPosition) -> Rect2i {
        Rect2i::new(
            self.config.px_per_field_edge as i32 * pos.col_in_img() as i32,
            self.config.px_per_field_edge as i32 * (pos.row_in_img() as i32 - 1),
            self.config.px_per_field_edge as _,
            2 * self.config.px_per_field_edge as i32,
        )
    }

    fn occupancy_on_field<M>(&self, mask: &M, pos: &FieldPosition) -> Result<f64>
    where
        M: MatTrait,
    {
        let mask_region = self.field_mask_roi(pos);
        let field = mask.roi(mask_region)?;

        // debug information
        if self.config.debug_field_config.matches(&pos) {
            use opencv::{
                core::Size,
                highgui::{imshow, wait_key_def},
                imgproc::{INTER_LINEAR, resize},
            };

            let mut tmp = Mat::default();
            mask.copy_to(&mut tmp)?;
            let mut field = tmp.roi_mut(mask_region)?;
            field.set_to_def(&Scalar::all(u8::MAX as f64))?;
            let mut out = Mat::default();
            resize(&tmp, &mut out, Size::default(), 0.5, 0.5, INTER_LINEAR)?;

            imshow(pos.to_string().as_str(), &out)?;
            wait_key_def()?;
        }

        let px_in_mask = count_non_zero(&field)?;
        let occupancy = px_in_mask as f64 / self.config.px_per_field() as f64;
        Ok(occupancy)
    }

    fn detect_pieces_on_field(&self, pos: &FieldPosition) -> Result<FieldOccupancy> {
        let occupancy_blacks = self.occupancy_on_field(&self.mask_blacks, pos)?;
        let occupancy_whites = 0.0; // TODO: occupancy_on_field(mask_whites, col, row, config)?;

        // can't have both at the same time, otherwise our colour mask is wrong
        assert!(occupancy_blacks < 0.5 || occupancy_whites < 0.5);

        let colour = if occupancy_blacks > 0.5 {
            PieceColour::Black
        } else if occupancy_whites > 0.5 {
            PieceColour::White
        } else {
            return Ok(FieldOccupancy::Empty);
        };

        Ok(FieldOccupancy::Occupied(colour, PieceType::Man))
    }
}

struct CalibrationData {
    // TODO: king threshold
}

pub struct CalibratedDetector {
    config: Config,
    _calibration_data: CalibrationData,
}

impl CalibratedDetector {
    pub fn detect_pieces<M>(&self, image: &M) -> Result<BoardLayout>
    where
        M: MatTrait + ToInputArray,
    {
        let detector = InitialImageProcessor::new(image, self.config)?;
        let detector = detector.process()?;
        detector.detect_pieces()
    }
}

pub struct Detector {
    config: Config,
}

impl Detector {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn calibrate(self) -> CalibratedDetector {
        CalibratedDetector {
            config: self.config,
            _calibration_data: CalibrationData {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_position_try_new() {
        let pos = FieldPosition::try_new(0, 0, 8).unwrap();
        assert_eq!("H1", pos.to_string());
    }

    #[test]
    fn test_field_position_try_from_img_pos() {
        let pos = FieldPosition::try_from_img_pos(2, 2, 8).unwrap();
        assert_eq!("H1", pos.to_string());
    }

    #[test]
    fn test_field_position_try_from_str() {
        let pos = FieldPosition::try_from_str("H1", 8).unwrap();
        assert_eq!("H1", pos.to_string());
    }

    #[test]
    fn test_field_position_row() {
        let pos = FieldPosition::try_new(3, 0, 8).unwrap();
        assert_eq!(3, pos.row());
        assert_eq!(5, pos.row_in_img());
    }

    #[test]
    fn test_field_position_col() {
        let pos = FieldPosition::try_new(0, 3, 8).unwrap();
        assert_eq!(3, pos.col());
        assert_eq!(5, pos.col_in_img());
    }

    #[test]
    fn test_field_iter() {
        let num_fields_per_line = 2;
        let mut board = BoardLayout::new(num_fields_per_line);
        board.set(
            &FieldPosition::try_new(0, 1, num_fields_per_line).unwrap(),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::Man),
        );
        board.set(
            &FieldPosition::try_new(1, 0, num_fields_per_line).unwrap(),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::King),
        );

        let mut fields = board.field_iter();
        assert_eq!(
            (
                FieldPosition::try_new(0, 0, num_fields_per_line).unwrap(),
                &FieldOccupancy::WrongType
            ),
            fields.next().unwrap()
        );
        assert_eq!(
            (
                FieldPosition::try_new(0, 1, num_fields_per_line).unwrap(),
                &FieldOccupancy::Occupied(PieceColour::Black, PieceType::Man)
            ),
            fields.next().unwrap()
        );
        assert_eq!(
            (
                FieldPosition::try_new(1, 0, num_fields_per_line).unwrap(),
                &FieldOccupancy::Occupied(PieceColour::Black, PieceType::King)
            ),
            fields.next().unwrap()
        );
        assert_eq!(
            (
                FieldPosition::try_new(1, 1, num_fields_per_line).unwrap(),
                &FieldOccupancy::WrongType
            ),
            fields.next().unwrap()
        );
        assert!(fields.next().is_none());
    }

    #[test]
    fn test_field_iter_blacks_only() {
        let num_fields_per_line = 2;
        let mut board = BoardLayout::new(num_fields_per_line);
        board.set(
            &FieldPosition::try_new(0, 1, num_fields_per_line).unwrap(),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::Man),
        );
        board.set(
            &FieldPosition::try_new(1, 0, num_fields_per_line).unwrap(),
            FieldOccupancy::Occupied(PieceColour::Black, PieceType::King),
        );

        let mut fields = board.field_iter().filter(|(pos, _)| pos.is_black_field());
        assert_eq!(
            (
                FieldPosition::try_new(0, 1, num_fields_per_line).unwrap(),
                &FieldOccupancy::Occupied(PieceColour::Black, PieceType::Man)
            ),
            fields.next().unwrap()
        );
        assert_eq!(
            (
                FieldPosition::try_new(1, 0, num_fields_per_line).unwrap(),
                &FieldOccupancy::Occupied(PieceColour::Black, PieceType::King)
            ),
            fields.next().unwrap()
        );
        assert!(fields.next().is_none());
    }
}
