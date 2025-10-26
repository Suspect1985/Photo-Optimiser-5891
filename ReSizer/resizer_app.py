"""
Re-Sizer - High-Performance Image Resizing and WebP Conversion Tool
Optimizes large image folders by resizing and converting to WebP format (lossy, quality 85).
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont

from PIL import Image
from PIL import ExifTags

# Try to import pillow-heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False


class ImageProcessor:
    """Handles image resizing and WebP conversion."""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    if HEIC_SUPPORT:
        SUPPORTED_FORMATS.add('.heic')

    MAX_DIMENSION = 3840

    @staticmethod
    def process_image(file_path: Path) -> Tuple[bool, str]:
        """
        Process a single image file.
        Returns: (success: bool, message: str)
        """
        try:
            with Image.open(file_path) as img:
                # Get original dimensions
                width, height = img.size
                max_dim = max(width, height)

                # Check if resizing is needed
                if max_dim <= ImageProcessor.MAX_DIMENSION:
                    return False, f"Skipped (already {max_dim}px): {file_path.name}"

                # Preserve EXIF data
                exif_data = None
                try:
                    exif_data = img.info.get('exif')
                except Exception:
                    pass

                # Calculate new dimensions (maintain aspect ratio)
                if width > height:
                    new_width = ImageProcessor.MAX_DIMENSION
                    new_height = int((height / width) * ImageProcessor.MAX_DIMENSION)
                else:
                    new_height = ImageProcessor.MAX_DIMENSION
                    new_width = int((width / height) * ImageProcessor.MAX_DIMENSION)

                # Resize image
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Convert RGBA to RGB if necessary (WebP can handle RGBA, but just in case)
                if img_resized.mode in ('RGBA', 'LA', 'P'):
                    # Keep RGBA for transparency support
                    pass
                elif img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')

                # Generate output path
                output_path = file_path.with_suffix('.webp')

                # Save as WebP (lossy, quality 85)
                save_kwargs = {
                    'format': 'WEBP',
                    'lossless': False,
                    'quality': 85
                }

                if exif_data:
                    save_kwargs['exif'] = exif_data

                img_resized.save(output_path, **save_kwargs)

                # Delete original file if conversion successful and filenames differ
                if output_path != file_path and output_path.exists():
                    file_path.unlink()

                return True, f"Resized {max_dim}px → {max(new_width, new_height)}px: {file_path.name}"

        except Exception as e:
            return False, f"Error processing {file_path.name}: {str(e)}"

    @staticmethod
    def find_images(root_folder: Path) -> List[Path]:
        """Recursively find all supported image files."""
        images = []
        for file_path in root_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ImageProcessor.SUPPORTED_FORMATS:
                images.append(file_path)
        return images


class ProcessingThread(QThread):
    """Background thread for image processing."""

    progress_update = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    finished = pyqtSignal(int, int, int)  # total_processed, total_resized, total_skipped

    def __init__(self, folder_path: str, max_workers: int = None):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.is_cancelled = False

    def run(self):
        """Execute the image processing workflow."""
        try:
            # Find all images
            self.log_message.emit(f"Scanning folder: {self.folder_path}")
            images = ImageProcessor.find_images(self.folder_path)
            total_files = len(images)

            if total_files == 0:
                self.log_message.emit("No supported images found.")
                self.finished.emit(0, 0, 0)
                return

            self.log_message.emit(f"Found {total_files} images. Starting processing...")
            self.log_message.emit(f"Using {self.max_workers} worker threads.")

            # Process images with thread pool
            total_resized = 0
            total_skipped = 0
            processed_count = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(ImageProcessor.process_image, img): img
                    for img in images
                }

                # Process results as they complete
                for future in as_completed(future_to_image):
                    if self.is_cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.log_message.emit("Processing cancelled.")
                        break

                    processed_count += 1

                    try:
                        was_resized, message = future.result()
                        self.log_message.emit(message)

                        if was_resized:
                            total_resized += 1
                        else:
                            total_skipped += 1

                    except Exception as e:
                        self.log_message.emit(f"Error: {str(e)}")
                        total_skipped += 1

                    # Update progress
                    self.progress_update.emit(processed_count, total_files)

            # Emit completion signal
            self.finished.emit(processed_count, total_resized, total_skipped)

        except Exception as e:
            self.log_message.emit(f"Fatal error: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            self.finished.emit(0, 0, 0)

    def cancel(self):
        """Cancel the processing."""
        self.is_cancelled = True


class ReSizerApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.processing_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Re-Sizer - Image Optimizer")
        self.setMinimumSize(800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title_label = QLabel("Re-Sizer")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel("Resize images > 3840px & convert to WebP (lossy, quality 85)")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)

        layout.addSpacing(20)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Folder:")
        folder_label.setMinimumWidth(60)
        folder_layout.addWidget(folder_label)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select a folder to process...")
        folder_layout.addWidget(self.folder_input)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folder)
        self.browse_button.setMinimumWidth(100)
        folder_layout.addWidget(self.browse_button)

        layout.addLayout(folder_layout)

        layout.addSpacing(10)

        # Start button
        self.start_button = QPushButton("Start Re-size")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.start_button)

        layout.addSpacing(10)

        # Progress bar
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Progress:")
        progress_layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        layout.addLayout(progress_layout)

        layout.addSpacing(10)

        # Status label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.status_label)

        layout.addSpacing(10)

        # Log output
        log_label = QLabel("Log:")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #f5f5f5; font-family: Consolas, monospace;")
        layout.addWidget(self.log_output)

        # Supported formats info
        formats_text = f"Supported: JPG, JPEG, PNG, TIFF, BMP{', HEIC' if HEIC_SUPPORT else ''}"
        formats_label = QLabel(formats_text)
        formats_label.setStyleSheet("color: #888; font-size: 10px;")
        formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(formats_label)

    def browse_folder(self):
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Process",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self.folder_input.setText(folder)

    def start_processing(self):
        """Start the image processing."""
        folder_path = self.folder_input.text().strip()

        if not folder_path:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return

        if not Path(folder_path).exists():
            QMessageBox.warning(self, "Invalid Folder", "The selected folder does not exist.")
            return

        # Disable controls
        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.folder_input.setEnabled(False)

        # Clear log
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running")
        self.status_label.setStyleSheet("font-weight: bold; color: #007acc;")

        # Start processing thread
        self.processing_thread = ProcessingThread(folder_path)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.log_message.connect(self.append_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.status_label.setText(f"Status: Running ({current}/{total})")

    def append_log(self, message: str):
        """Append message to log output."""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def processing_finished(self, total_processed: int, total_resized: int, total_skipped: int):
        """Handle processing completion."""
        # Re-enable controls
        self.start_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.folder_input.setEnabled(True)

        # Update status
        self.status_label.setText("Status: Complete")
        self.status_label.setStyleSheet("font-weight: bold; color: #28a745;")

        # Summary
        summary = f"\n{'='*50}\n"
        summary += "Re-sizing Complete!\n"
        summary += f"{'='*50}\n"
        summary += f"Total Processed: {total_processed}\n"
        summary += f"Total Resized: {total_resized}\n"
        summary += f"Total Skipped: {total_skipped}\n"
        self.append_log(summary)

        # Show completion dialog
        QMessageBox.information(
            self,
            "Complete",
            f"Re-sizing complete!\n\n"
            f"Processed: {total_processed}\n"
            f"Resized: {total_resized}\n"
            f"Skipped: {total_skipped}"
        )

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Processing is still running. Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.processing_thread.cancel()
                self.processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern cross-platform style

    window = ReSizerApp()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
