# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os
import Qt.QtWidgets as qtw
from chimerax.ui.gui import MainToolWindow
from chimerax.core.tools import ToolInstance
from Qt.QtWidgets import QLabel
from Qt import QtWidgets
from Qt.QtWidgets import QGridLayout, QFileDialog
from chimerax.core.commands import run


class DLStruct(ToolInstance):
    """
    DeepSSETracer GUI for ChimeraX.
    Provides interface for secondary structure element prediction from cryo-EM maps.
    """

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:user/tools/tutorial.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.cwd = os.getcwd()
        self.display_name = "DeepSSETracer 1.1"
        self.tool_window = MainToolWindow(self)
        self._build_ui()

    def _build_ui(self):
        """Build the main GUI interface."""
        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # MRC input row
        self.mrc_label = QLabel("MRC:")
        self.mrc_text = qtw.QLineEdit()
        self.mrc_button = qtw.QPushButton("Browse")
        
        tooltip_mrc = ("Path to MRC file of the density map. "
                       "Works best for 5-10Å resolution maps.")
        self.mrc_label.setToolTip(tooltip_mrc)
        self.mrc_text.setToolTip(tooltip_mrc)
        self.mrc_button.setToolTip(tooltip_mrc)

        # Output directory row
        self.out_label = QLabel("Output Dir:")
        self.out_text = qtw.QLineEdit()
        self.out_button = qtw.QPushButton("Browse")
        
        tooltip_out = "Directory to save prediction results."
        self.out_text.setToolTip(tooltip_out)
        self.out_button.setToolTip(tooltip_out)
        
        # Run button
        self.run_button = qtw.QPushButton("Run")
        self.run_button.setToolTip(
            "Start prediction. Processing time: ~7s with GPU, ~80s with CPU "
            "(for 74×59×83 map)."
        )

        # Layout assembly
        row = 0
        self.layout.addWidget(self.mrc_label, row, 0)
        self.layout.addWidget(self.mrc_text, row, 1)
        self.layout.addWidget(self.mrc_button, row, 2)
        self.mrc_button.clicked.connect(self.browse_mrc)

        row += 1
        self.layout.addWidget(self.out_label, row, 0)
        self.layout.addWidget(self.out_text, row, 1)
        self.layout.addWidget(self.out_button, row, 2)
        self.out_button.clicked.connect(self.browse_out)

        row += 1
        self.layout.addWidget(self.run_button, row, 1)
        self.run_button.clicked.connect(self.submit_job)

        self.tool_window.ui_area.setLayout(self.layout)
        self.tool_window.manage('side')

    def browse_out(self):
        """Open directory browser for output location."""
        dir_out = QtWidgets.QFileDialog.getExistingDirectory(caption="Output Directory")
        if dir_out:
            self.out_text.setText(dir_out)
            print(f"\nOutput directory: {dir_out}")
        else:
            print("\nSelection cancelled")

    def resample_map(self, input_path, output_dir):
        """Resample density map to 1Å voxel spacing."""
        from os import path
        
        map_model = run(self.session, f'open {input_path}')[0]
        map_1A = run(self.session, f'volume resample #{map_model.id_string} spacing 1.0')
        
        filename = path.split(input_path)[1]
        basename = path.splitext(filename)[0]
        output_path = path.join(output_dir, f'{basename}_1A.mrc')
        
        run(self.session, f'save {output_path} model #{map_1A.id_string}')
        return output_path

    def submit_job(self):
        """Execute SSE prediction pipeline."""
        import torch
        from . import deepssetracer
        import os
        from chimerax.map_data import mrc

        output_dir = self.out_text.text()
        map_path = self.mrc_text.text()
        
        # Check voxel size and resample if needed
        voxel_size = mrc.open(map_path)[0].step
        print(f"Input map voxel size: {voxel_size[0]:.2f} × {voxel_size[1]:.2f} × {voxel_size[2]:.2f} Å")
        
        if voxel_size[0] != 1 or voxel_size[1] != 1 or voxel_size[2] != 1:
            print("Resampling to 1Å voxel spacing...")
            map_path = self.resample_map(map_path, output_dir)

        # Configure prediction parameters
        class Args:
            cuda = torch.cuda.is_available()
            seed = 0
            mrc_path = map_path
            pred_helix_path = os.path.join(output_dir, 'pre_helix.mrc')
            pred_sheet_path = os.path.join(output_dir, 'pre_sheet.mrc')
            pred_helix_path_NoEdge = os.path.join(output_dir, 'pre_helix_Cropped.mrc')
            pred_sheet_path_NoEdge = os.path.join(output_dir, 'pre_sheet_Cropped.mrc')
            layers = 3
            output_path = output_dir

        args = Args()
        deepssetracer.predict_secondary_structure(args)
        
        # Open predicted maps in ChimeraX
        for path in [args.pred_helix_path, args.pred_sheet_path,
                     args.pred_helix_path_NoEdge, args.pred_sheet_path_NoEdge]:
            run(self.session, f'open {path}')

    def browse_mrc(self):
        """Open file browser for MRC map selection."""
        fname, _ = QFileDialog.getOpenFileName(
            caption="Select MRC File",
            filter="MRC Files (*.mrc *.map *.ccp4)"
        )
        if fname:
            self.mrc_text.setText(fname)
            print(f"\nSelected MRC: {fname}")
        else:
            print("\nSelection cancelled")

