from PyQt5.QtWidgets import QLineEdit, QLabel

from redundant.base_dialog import BaseDialog

class FilteringDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(title="Filtering Condition", parent=parent)

    def setup_ui(self):
        super().setup_ui()

        # Add specific input fields
        self.condition_input = QLineEdit(self)
        self.layout.insertWidget(0, QLabel("Filtering Condition:", self))
        self.layout.insertWidget(1, self.condition_input)

    def validate_params(self) -> bool:
        # Validate the subsampling rate
        try:
            condition = self.condition_input.text()
            self.params["condition"] = condition
            return True
        except ValueError:
            return False
