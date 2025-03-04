import sys
from PyQt5.QtWidgets import QApplication
from gui.dialog_boxes.subtract_dialog import LogicalOperationsDialog

def test_combo_box_dialog():
    app = QApplication(sys.argv)

    # Define options for the combo box
    options = ["Option 1", "Option 2", "Option 3", "Option 4"]

    # Create and display the dialog
    dialog = LogicalOperationsDialog("Select an Option", options)
    if dialog.exec_() == dialog.Accepted:
        selected_params = dialog.get_parameters()
        print("Selected Option:", selected_params["selected_option"])
    else:
        print("Dialog was cancelled.")

    sys.exit(app.exec_())

if __name__ == "__main__":
    test_combo_box_dialog()
