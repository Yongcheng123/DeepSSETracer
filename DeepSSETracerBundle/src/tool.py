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
    # Inheriting from ToolInstance makes us known to the ChimeraX tool mangager,
    # so we can be notified and take appropriate action when sessions are closed,
    # saved, or restored, and we will be listed among running tools and so on.
    #
    # If cleaning up is needed on finish, override the 'delete' method
    # but be sure to call 'delete' from the superclass at the end.

    SESSION_ENDURING = False    # Does this instance persist when session closes
    SESSION_SAVE = True         # We do save/restore in sessions
    help = "help:user/tools/tutorial.html"
                                # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        # 'session'   - chimerax.core.session.Session instance
        # 'tool_name' - string

        # Initialize base class.
        super().__init__(session, tool_name)
        self.cwd = os.getcwd()

        # Set name displayed on title bar (defaults to tool_name)
        # Must be after the superclass init, which would override it.
        self.display_name = "DeepSSETracer 1.1"

        # Create the main window for our tool.  The window object will have
        # a 'ui_area' where we place the widgets composing our interface.
        # The window isn't shown until we call its 'manage' method.
        #
        # Note that by default, tool windows are only hidden rather than
        # destroyed when the user clicks the window's close button.  To change
        # this behavior, specify 'close_destroys=True' in the MainToolWindow
        # constructor.

        self.tool_window = MainToolWindow(self)

        # We will be adding an item to the tool's context menu, so override
        # the default MainToolWindow fill_context_menu method
        # self.tool_window.fill_context_menu = self.fill_context_menu

        # Our user interface is simple enough that we could probably inline
        # the code right here, but for any kind of even moderately complex
        # interface, it is probably better to put the code in a method so
        # that this __init__ method remains readable.
        self._build_ui()

    def _build_ui(self):
        # Put our widgets in the tool window

        # We will use an editable single-line text input field (QLineEdit)
        # with a descriptive text label to the left of it (QLabel).  To
        # arrange them horizontally side by side we use QHBoxLayout

        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.mrc_label = QLabel("MRC:")
        self.mrc_text = qtw.QLineEdit()
        self.mrc_button = qtw.QPushButton("Browse")
        self.mrc_label.setToolTip("Enter the path to an MRC file of the map to predict \n"
                                  "alpha-helix and beta-sheet locations with DeepSSETracer. \n"
                                  "It works best for map resolutions from 5-10 Angstroms.")
        self.mrc_text.setToolTip("Enter the path to an MRC file of the map.")
        self.mrc_button.setToolTip("Enter the path to an MRC file of the map.")

        self.out_label = QLabel("Output Dir:")
        self.out_text = qtw.QLineEdit()
        self.out_button = qtw.QPushButton("Browse")
        self.out_text.setToolTip("Enter the path to an folder to save the prediction results.")
        self.out_button.setToolTip("Enter the path to an folder to save the prediction results.")
        self.run_button = qtw.QPushButton("Run")
        self.run_button.setToolTip("The procedure will take a few seconds to minutes. \n"
                                   "Example: For a map of size 74*59*83 this might take \n"
                                   "7 seconds with a GPU, and 81 seconds without a GPU.")

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

        # Set the layout as the contents of our window
        self.tool_window.ui_area.setLayout(self.layout)

        # Show the window on the user-preferred side of the ChimeraX
        # main window
        self.tool_window.manage('side')

    def browse_out(self):
        dir_out = QtWidgets.QFileDialog.getExistingDirectory(caption="Output Directory")
        if dir_out == "":
            print("\nCancel")
            return
        self.out_text.setText(dir_out)
        print("\nOutput Directory path: ")
        print(dir_out)

    def resample_map(self, input_map_path, output_directory):
        from os import path
        # Open the input map
        map = run(self.session, f'open {input_map_path}')[0]
        # Resample it to 1A grid spacing
        map_1A = run(self.session, f'volume resample #{map.id_string} spacing 1.0')
        # Figure out path to save new 1A file
        map_filename = path.split(input_map_path)[1]
        map_basename = path.splitext(map_filename)[0]
        map_1A_path = path.join(output_directory, map_basename + '_1A.mrc')
        # Save new 1A map
        run(self.session, f'save {map_1A_path} model #{map_1A.id_string}')
        # self.session.models.close([map, map_1A])
        return map_1A_path

    def submit_job(self):
        import torch
        from . import deepssetracer
        import os
        from chimerax.map_data import mrc, ArrayGridData

        output = self.out_text.text()
        map_path = self.mrc_text.text()
        voxel_size_x, voxel_size_y, voxel_size_z = mrc.open(map_path)[0].step
        print("Input map voxel size: {0} - {1} - {2}...".format(voxel_size_x, voxel_size_y, voxel_size_z))
        if voxel_size_x != 1 or voxel_size_y != 1 or voxel_size_z != 1:
            print("Resampling voxel size to 1A...")
            map_1A_path = self.resample_map(input_map_path=map_path, output_directory=output)
        else:
            map_1A_path = map_path

        class Args:
            cuda = "yes"
            seed = 0
            mrc_path = map_1A_path
            pred_helix_path = output + os.path.sep + 'pre_helix.mrc'
            pred_sheet_path = output + os.path.sep + 'pre_sheet.mrc'
            pred_helix_path_NoEdge = output + os.path.sep + 'pre_helix_Cropped.mrc'
            pred_sheet_path_NoEdge = output + os.path.sep + 'pre_sheet_Cropped.mrc'
            layers = 3
            output_path = output

        print("\ncwd: ", os.getcwd())
        args = Args()
        args.cuda = True if args.cuda == "yes" else False
        if args.cuda is True and not torch.cuda.is_available():
            print("\nCUDA not available, disabling...")
            args.cuda = False
        deepssetracer.deepssetracer_model(args)
        run(self.session, 'open ' + args.pred_helix_path)
        run(self.session, 'open ' + args.pred_sheet_path)
        run(self.session, 'open ' + args.pred_helix_path_NoEdge)
        run(self.session, 'open ' + args.pred_sheet_path_NoEdge)

    def browse_mrc(self):
        fname_mrc, _ = QFileDialog.getOpenFileName(caption="MRC", filter="MRC File (*.mrc *.map *.ccp4)")
        if fname_mrc == "":
            print("\nCancel")
            return
        self.mrc_text.setText(fname_mrc)
        print("\nMRC path: ")
        print(fname_mrc)

    def return_pressed(self):
        # The use has pressed the Return key; log the current text as HTML
        from chimerax.core.commands import run
        # ToolInstance has a 'session' attribute...
        run(self.session, "log html %s" % self.line_edit.text())

    def fill_context_menu(self, menu, x, y):
        # Add any tool-specific items to the given context menu (a QMenu instance).
        # The menu will then be automatically filled out with generic tool-related actions
        # (e.g. Hide Tool, Help, Dockable Tool, etc.) 
        #
        # The x,y args are the x() and y() values of QContextMenuEvent, in the rare case
        # where the items put in the menu depends on where in the tool interface the menu
        # was raised.
        from Qt.QtWidgets import QAction
        clear_action = QAction("Clear", menu)
        clear_action.triggered.connect(lambda *args: self.line_edit.clear())
        menu.addAction(clear_action)

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'current text': self.line_edit.text()
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        # Instead of using a fixed string when calling the constructor below, we could
        # have saved the tool name during take_snapshot() (from self.tool_name, inherited
        # from ToolInstance) and used that saved tool name.  There are pros and cons to
        # both approaches.
        inst = class_obj(session, "Tutorial (Qt)")
        inst.line_edit.setText(data['current text'])
        return inst
