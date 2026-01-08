from PyQt5.QtWidgets import QLineEdit, QLabel

from redundant.base_dialog import BaseDialog

class SubsamplingDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(title="Subsampling Parameters", parent=parent)

    def setup_ui(self):
        super().setup_ui()

        # Add specific input fields
        self.rate_input = QLineEdit(self)
        self.layout.insertWidget(0, QLabel("Subsampling Rate:", self))
        self.layout.insertWidget(1, self.rate_input)

    def validate_params(self) -> bool:
        # Validate the subsampling rate
        try:
            rate = float(self.rate_input.text())
            if rate <= 0 or rate > 1:
                raise ValueError
            self.params["rate"] = rate
            return True
        except ValueError:
            return False
