//! Helpers to init specific cameras (if needed).

use log::debug;
use url::Url;

/// Represents an ESP32-CAM.
#[derive(Debug)]
pub struct Esp32Cam;

impl Esp32Cam {
    fn set_max_resolution(base_url: &Url) {
        let resolution_config_url = base_url.join("control?var=framesize&val=15").unwrap();
        reqwest::blocking::get(resolution_config_url).unwrap();
    }

    fn set_max_quality(base_url: &Url) {
        let resolution_config_url = base_url.join("control?var=quality&val=63").unwrap();
        reqwest::blocking::get(resolution_config_url).unwrap();
    }

    /// Set the required parameters on the ESP32-CAM to acquire images as needed by the application.
    pub fn init(base_url: &Url) {
        debug!("ESP32-CAM: setting max. resolution & quality");
        Self::set_max_resolution(base_url);
        Self::set_max_quality(base_url);
    }
}
