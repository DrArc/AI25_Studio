from PyQt6.QtWidgets import QFileDialog, QMessageBox

def capture_viewport(viewer, parent=None):
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Screenshot",
        "",
        "PNG Image (*.png);;JPEG Image (*.jpg)"
    )

    if file_path:
        try:
            viewer._display.View.Dump(file_path)
            QMessageBox.information(parent, "Success", f"Viewport captured to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(parent, "Error", f"Capture failed:\n{e}")
