from PyQt5.QtWidgets import QLineEdit, QLabel

from redundant.base_dialog import BaseDialog


class DbscanDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(title="Dbscan Parameters", parent=parent)

    def setup_ui(self):
        super().setup_ui()

        # Add specific input fields
        self.eps_input = QLineEdit(self)
        self.min_samples_input = QLineEdit(self)
        self.layout.insertWidget(0, QLabel("Epsilon:", self))
        self.layout.insertWidget(1, self.eps_input)
        self.layout.insertWidget(2, QLabel("Min Samples:", self))
        self.layout.insertWidget(3, self.min_samples_input)

    def validate_params(self) -> bool:
        # Validate clustering parameters
        try:
            eps = float(self.eps_input.text())
            min_samples = int(self.min_samples_input.text())
            if eps <= 0 or min_samples <= 0:
                raise ValueError
            self.params["eps"] = eps
            self.params["min_samples"] = min_samples
            return True
        except ValueError:
            return False
