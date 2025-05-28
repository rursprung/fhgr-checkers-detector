use crate::detector::{
    self, CalibrationData, DebugFieldConfig, FieldOccupancy, FieldPosition, UncalibratedDetector,
};
use Error::*;
use opencv::{core::ToInputArray, prelude::*};
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum Error {
    MissingPieces,
    InvalidPieces,
    InternalDetectionError(detector::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MissingPieces => write!(f, "Not all pieces detected in the expected locations"),
            InvalidPieces => write!(
                f,
                "Did not detect the expected pieces in the correct places"
            ),
            InternalDetectionError(_) => write!(f, "internal detection error"),
        }
    }
}

impl From<detector::Error> for Error {
    fn from(e: detector::Error) -> Self {
        InternalDetectionError(e)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            InternalDetectionError(e) => Some(e),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Config {
    pub num_fields_per_line: u8,
    pub px_per_field_edge: u8,
    pub debug_field: DebugFieldConfig,
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct ReferencePositions {
    top_man: FieldPosition,
    top_king: FieldPosition,
    bottom_man: FieldPosition,
    bottom_king: FieldPosition,
}

impl Display for ReferencePositions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "top row: man on {}, king on {}, ",
            self.top_man, self.top_king
        )?;
        write!(
            f,
            "bottom row: man on {}, king on {}",
            self.bottom_man, self.bottom_king
        )?;
        Ok(())
    }
}

pub fn reference_positions(config: &Config) -> ReferencePositions {
    let is_even_number_of_fields = config.num_fields_per_line % 2 == 0;
    let max_col_on_first_row =
        config.num_fields_per_line - 1 - if is_even_number_of_fields { 0 } else { 1 };
    let max_col_on_last_row =
        config.num_fields_per_line - 1 - if is_even_number_of_fields { 1 } else { 0 };
    ReferencePositions {
        top_man: FieldPosition::try_new(0, 1, config.num_fields_per_line).unwrap(),
        top_king: FieldPosition::try_new(0, max_col_on_first_row, config.num_fields_per_line)
            .unwrap(),
        bottom_man: FieldPosition::try_new(7, 0, config.num_fields_per_line).unwrap(),
        bottom_king: FieldPosition::try_new(7, max_col_on_last_row, config.num_fields_per_line)
            .unwrap(),
    }
}

fn field_height_or_none(field_occupancy: &FieldOccupancy) -> Option<u32> {
    match field_occupancy {
        FieldOccupancy::Occupied(_, _, height) => Some(*height),
        _ => None,
    }
}

pub fn calibrate<M>(image: &M, config: &Config) -> Result<CalibrationData, Error>
where
    M: MatTrait + ToInputArray,
{
    let detector = UncalibratedDetector::new(crate::detector::Config {
        num_fields_per_line: config.num_fields_per_line,
        px_per_field_edge: config.px_per_field_edge,
        debug_field: config.debug_field,
    });
    let result = detector.detect_pieces(image)?;

    let reference_positions = reference_positions(config);

    let top_man_height = result
        .get(&reference_positions.top_man)
        .map_or(None, field_height_or_none);
    let top_king_height = result
        .get(&reference_positions.top_king)
        .map_or(None, field_height_or_none);
    let bottom_man_height = result
        .get(&reference_positions.bottom_man)
        .map_or(None, field_height_or_none);
    let bottom_king_height = result
        .get(&reference_positions.bottom_king)
        .map_or(None, field_height_or_none);

    if top_man_height.is_none()
        || top_king_height.is_none()
        || bottom_man_height.is_none()
        || bottom_king_height.is_none()
    {
        return Err(MissingPieces);
    }
    let top_man_height = top_man_height.unwrap();
    let top_king_height = top_king_height.unwrap();
    let bottom_man_height = bottom_man_height.unwrap();
    let bottom_king_height = bottom_king_height.unwrap();

    if bottom_king_height >= top_king_height
        || bottom_man_height >= top_man_height
        || top_man_height >= top_king_height
        || bottom_man_height >= bottom_king_height
    {
        return Err(InvalidPieces);
    }

    let top_height_diff = top_king_height - top_man_height;
    let bottom_height_diff = bottom_king_height - bottom_man_height;

    // presume linear distribution
    let king_threshold_height_on_top_row = top_man_height + top_height_diff / 2;
    let king_threshold_height_on_bottom_row = bottom_man_height + bottom_height_diff / 2;
    let king_threshold_diff_per_field = (king_threshold_height_on_top_row
        - king_threshold_height_on_bottom_row)
        / config.num_fields_per_line as u32;

    Ok(CalibrationData {
        king_threshold_height_on_top_row,
        king_threshold_diff_per_field: king_threshold_diff_per_field,
    })
}

pub fn try_calibrate<M>(image: &M, config: &Config) -> Result<Option<CalibrationData>, Error>
where
    M: MatTrait + ToInputArray,
{
    match calibrate(image, config) {
        Ok(calibration) => Ok(Some(calibration)),
        Err(MissingPieces | InvalidPieces) => Ok(None),
        Err(e) => Err(e),
    }
}
