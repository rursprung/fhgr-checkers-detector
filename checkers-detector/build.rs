fn main() {
    #[cfg(feature = "resolve_opencv_with_vcpkg")]
    vcpkg::find_package("opencv4").unwrap();
}
