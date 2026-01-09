#!/usr/bin/env python3
"""
CSV Grid Plotter with per-graph configs and movable crosshair.

Features:
- Load CSV files; each column (except time) stored as variable named: <col>_<filename>
- Grid layout for graphs (default 3x3). You can choose which cell to place graphs.
- Each GraphWidget has toolbar: Add Var, Remove Var, Config Var, Move Graph, Remove Graph, Zoom In/Out, Auto.
- Crosshair mode: enable a vertical movable line that displays values of all variables in that graph at the nearest time.
- Per-variable config: time_start, time_end, scale, color, plot_type (line/scatter).
"""

import sys
import os
import pandas as pd
import numpy as np
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QLabel, QMessageBox, QGridLayout, QDialog,
    QFormLayout, QLineEdit, QColorDialog, QComboBox, QInputDialog, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg


# -------------------------
# Variable configuration dialog
# -------------------------
class VariableConfigDialog(QDialog):
    def __init__(self, var_name, time_min, time_max, existing_cfg=None):
        super().__init__()
        self.setWindowTitle(f"Configure {var_name}")
        self.setMinimumWidth(300)

        self.color = QColor(existing_cfg["color"]) if existing_cfg else QColor("blue")

        layout = QFormLayout(self)
        self.time_start = QLineEdit(str(existing_cfg["time_start"] if existing_cfg else float(time_min)))
        self.time_end = QLineEdit(str(existing_cfg["time_end"] if existing_cfg else float(time_max)))
        self.scale = QLineEdit(str(existing_cfg["scale"] if existing_cfg else 1.0))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["line", "scatter"])
        if existing_cfg:
            self.plot_type.setCurrentText(existing_cfg["plot_type"])

        self.color_btn = QPushButton("Choose Color")
        self.color_btn.clicked.connect(self.pick_color)
        self.color_btn.setStyleSheet(f"background-color: {self.color.name()};")

        layout.addRow("Start Time:", self.time_start)
        layout.addRow("End Time:", self.time_end)
        layout.addRow("Scale Factor:", self.scale)
        layout.addRow("Plot Type:", self.plot_type)
        layout.addRow("Color:", self.color_btn)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

    def pick_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Plot Color")
        if color.isValid():
            self.color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")

    def get_config(self):
        return {
            "time_start": float(self.time_start.text()),
            "time_end": float(self.time_end.text()),
            "scale": float(self.scale.text()),
            "color": self.color.name(),
            "plot_type": self.plot_type.currentText()
        }


# Dialog to create a computed variable from expression of existing variables
class ComputedVarDialog(QDialog):
    def __init__(self, variables):
        super().__init__()
        self.setWindowTitle("Create Computed Variable")
        self.setMinimumWidth(420)
        self.variables = variables
        # Use a form layout but include a variable-list to help building expressions
        layout = QFormLayout()

        # name for the result variable
        self.name_edit = QLineEdit("computed")

        # expression text
        self.expr_edit = QLineEdit()
        hint = QLabel("Double-click a variable to insert into the expression. Example: +, -, *, / and constants are allowed.")
        hint.setWordWrap(True)

        # variable list (user can double-click to insert)
        self.var_list = QListWidget()
        for n in sorted(self.variables.keys()):
            self.var_list.addItem(QListWidgetItem(n))
        self.var_list.itemDoubleClicked.connect(lambda it: self.insert_var(it.text()))

        insert_btn = QPushButton("Insert >>")
        insert_btn.clicked.connect(lambda: self._insert_selected_var())

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("Variables:"))
        right_box.addWidget(self.var_list)
        right_box.addWidget(insert_btn)

        # choose time base for interpolation
        self.time_base = QComboBox()
        for n in sorted(self.variables.keys()):
            self.time_base.addItem(n)

        # additional numeric adjustments: zero (baseline subtract) and offset (add after scale)
        self.zero_edit = QLineEdit("0.0")
        self.offset_edit = QLineEdit("0.0")

        # reuse same config fields as VariableConfigDialog
        self.time_start = QLineEdit("")
        self.time_end = QLineEdit("")
        self.scale = QLineEdit("1.0")
        self.plot_type = QComboBox()
        self.plot_type.addItems(["line", "scatter"])
        self.color = QColor("#0000ff")
        self.color_btn = QPushButton("Choose Color")
        self.color_btn.clicked.connect(self.pick_color)
        self.color_btn.setStyleSheet(f"background-color: {self.color.name()};")

        # build layouts
        top_h = QHBoxLayout()
        left_v = QVBoxLayout()
        left_v.addWidget(QLabel("Result name:"))
        left_v.addWidget(self.name_edit)
        left_v.addWidget(QLabel("Expression:"))
        left_v.addWidget(self.expr_edit)
        left_v.addWidget(hint)
        top_h.addLayout(left_v, 3)
        top_h.addLayout(right_box, 2)

        layout.addRow(top_h)
        layout.addRow("Time base (interpolate others):", self.time_base)
        layout.addRow("Zero (subtract):", self.zero_edit)
        layout.addRow("Offset (add):", self.offset_edit)
        layout.addRow("Start Time (optional):", self.time_start)
        layout.addRow("End Time (optional):", self.time_end)
        layout.addRow("Scale:", self.scale)
        layout.addRow("Plot Type:", self.plot_type)
        layout.addRow("Color:", self.color_btn)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

        self.setLayout(layout)

    def pick_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Plot Color")
        if color.isValid():
            self.color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")

    def insert_var(self, var_name: str):
        # insert var name into expression at cursor
        if var_name:
            self.expr_edit.insert(var_name)

    def _insert_selected_var(self):
        it = self.var_list.currentItem()
        if it:
            self.insert_var(it.text())

    def get_result(self):
        # return (name, time_array, value_array, cfg_dict, expr_text, time_base)
        return {
            "name": self.name_edit.text().strip(),
            "expr": self.expr_edit.text().strip(),
            "time_base": self.time_base.currentText(),
            "zero": float(self.zero_edit.text() or 0.0),
            "offset": float(self.offset_edit.text() or 0.0),
            "time_start": float(self.time_start.text()) if self.time_start.text() else None,
            "time_end": float(self.time_end.text()) if self.time_end.text() else None,
            "scale": float(self.scale.text() or 1.0),
            "color": self.color.name(),
            "plot_type": self.plot_type.currentText()
        }


# -------------------------
# Graph widget
# -------------------------
class GraphWidget(QWidget):
    def __init__(self, grid_manager, row, col, grid_cell_count):
        super().__init__()
        self.grid_manager = grid_manager
        self.row = row
        self.col = col
        self.grid_cell_count = grid_cell_count

        self.variables = None  # reference to global variables dict
        self.curves = {}       # var_name -> curve
        self.var_configs = {}  # per-graph per-var config
        self.pick_enabled = False

        # UI
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()

        self.add_btn = QPushButton("➕ Add Var")
        self.add_comp_btn = QPushButton("➕ Add Comp Var")
        self.rem_btn = QPushButton("➖ Remove Var")
        self.cfg_btn = QPushButton("⚙️ Config Var")
        self.move_btn = QPushButton("🔀 Move Graph")
        self.del_btn = QPushButton("🗑️ Remove Graph")
        self.zoom_in_btn = QPushButton("🔍 +")
        self.zoom_out_btn = QPushButton("🔍 -")
        self.auto_btn = QPushButton("Auto")
        self.pick_toggle_btn = QPushButton("🔎 Crosshair OFF")

        for b in (self.add_btn, self.add_comp_btn, self.rem_btn, self.cfg_btn, self.move_btn,
              self.del_btn, self.zoom_in_btn, self.zoom_out_btn, self.auto_btn, self.pick_toggle_btn):
            toolbar.addWidget(b)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.setBackground("w")
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Value")
        layout.addWidget(self.plot)

        # Crosshair objects (vertical line + text)
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('k', style=Qt.DashLine))
        self.vline.setZValue(1000)
        self.text_item = pg.TextItem("", anchor=(0, 1), fill=(255,255,255,200))
        
        # initially not added to plot

        # connections
        self.add_btn.clicked.connect(self.add_var_dialog)
        self.add_comp_btn.clicked.connect(self.add_computed_var_dialog)
        self.rem_btn.clicked.connect(self.remove_var_dialog)
        self.cfg_btn.clicked.connect(self.config_var_dialog)
        self.move_btn.clicked.connect(self.move_graph_dialog)
        self.del_btn.clicked.connect(self.remove_self)
        self.zoom_in_btn.clicked.connect(lambda: self.plot.getViewBox().scaleBy((0.8, 0.8)))
        self.zoom_out_btn.clicked.connect(lambda: self.plot.getViewBox().scaleBy((1.2, 1.2)))
        self.auto_btn.clicked.connect(self.safe_enable_autorange)
        self.pick_toggle_btn.clicked.connect(self.toggle_pick_mode)
        self.vline.sigPositionChanged.connect(self.crosshair_moved)

    def safe_enable_autorange(self):
        """Enable autorange without including the crosshair line."""
        was_visible = self.pick_enabled and (self.vline in self.plot.items())
        if was_visible:
            self.plot.removeItem(self.vline)
        self.plot.enableAutoRange()
        if was_visible:
            self.plot.addItem(self.vline)

        # For clicking/dragging events

    def set_variables_source(self, variables):
        self.variables = variables

    # Add variable to this graph (with dialog)
    def add_var_dialog(self):
        if not self.variables:
            QMessageBox.warning(self, "No variables", "Load CSVs first to have variables.")
            return
        names = list(self.variables.keys())
        var_name, ok = QInputDialog.getItem(self, "Select variable", "Variable:", names, 0, False)
        if ok and var_name:
            self.add_variable(var_name)

    def add_computed_var_dialog(self):
        if not self.variables:
            QMessageBox.warning(self, "No variables", "Load CSVs first to have variables.")
            return
        dlg = ComputedVarDialog(self.variables)
        if dlg.exec_() != QDialog.Accepted:
            return
        res = dlg.get_result()
        name = res["name"] or "computed"
        expr = res["expr"]
        time_base = res["time_base"]
        if not expr:
            QMessageBox.warning(self, "Empty expression", "Please enter an expression or insert variables from the list.")
            return
        if time_base not in self.variables:
            QMessageBox.warning(self, "Invalid time base", "Selected time base variable not available.")
            return
        try:
            base_x = np.array(self.variables[time_base][0], dtype=float)
            # build namespace: interpolate all variables onto base_x
            ns = {}
            for k, (x, y) in self.variables.items():
                if x is None or y is None:
                    raise ValueError(f"Variable {k} has no data")
                x_arr = np.array(x, dtype=float)
                y_arr = np.array(y, dtype=float)
                if k == time_base:
                    ns[k] = y_arr
                else:
                    ns[k] = np.interp(base_x, x_arr, y_arr)
            # preprocess expression: replace caret '^' with '**' (common user input),
            # remove commas inside numbers (e.g., 1,234.56) which cause decimal literal errors
            expr_safe = expr.replace('^', '**')
            expr_safe = re.sub(r'(?<=\d),(?=\d)', '', expr_safe)

            # Create safe python identifiers for variables (some variable names may contain
            # characters invalid in Python identifiers). Replace occurrences in the expression
            # with these temporary names.
            name_map = {}
            eval_ns = {}
            # Add safe numpy functions
            safe_funcs = {
                'np': np, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs
            }
            eval_ns.update(safe_funcs)

            # sort variable names by length desc to avoid partial-substitution issues
            for idx, orig in enumerate(sorted(ns.keys(), key=lambda s: -len(s))):
                safe_name = f"_v{idx}"
                name_map[orig] = safe_name
                eval_ns[safe_name] = ns[orig]
                # replace occurrences of orig in the expression with safe_name
                # use lookarounds to avoid replacing inside other words
                pattern = rf'(?<![\w]){re.escape(orig)}(?![\w])'
                expr_safe = re.sub(pattern, safe_name, expr_safe)

            # evaluate expression in restricted namespace
            y = eval(expr_safe, {"__builtins__": None}, eval_ns)
            y = np.array(y, dtype=float)
            # apply zero/scale/offset
            y = (y - float(res.get("zero", 0.0))) * float(res.get("scale", 1.0)) + float(res.get("offset", 0.0))

            # register variable globally on grid manager
            gm = self.grid_manager
            final_name = name
            if final_name in gm.variables:
                idx = 1
                while f"{final_name}_{idx}" in gm.variables:
                    idx += 1
                final_name = f"{final_name}_{idx}"
            gm.variables[final_name] = (base_x, y)
            gm.var_list.addItem(QListWidgetItem(final_name))

            # config for plotting (use provided time_start/end or full range)
            cfg = {
                "time_start": float(res["time_start"]) if res["time_start"] is not None else float(base_x[0]),
                "time_end": float(res["time_end"]) if res["time_end"] is not None else float(base_x[-1]),
                "scale": 1.0,
                "color": res["color"],
                "plot_type": res["plot_type"]
            }
            self.var_configs[final_name] = cfg
            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "Expression error", f"Failed to evaluate expression:\n{e}")

    def add_variable(self, var_name):
        if var_name in self.var_configs:
            QMessageBox.information(self, "Already added", f"{var_name} already added to this graph.")
            return
        x, y = self.variables[var_name]
        cfg = {
            "time_start": float(x[0]),
            "time_end": float(x[-1]),
            "scale": 1.0,
            "color": "#0000ff",
            "plot_type": "line"
        }
        # open config dialog to let user tweak now
        dlg = VariableConfigDialog(var_name, x[0], x[-1], existing_cfg=cfg)
        if dlg.exec_() == QDialog.Accepted:
            cfg = dlg.get_config()
        self.var_configs[var_name] = cfg
        self.update_plot()

    def remove_var_dialog(self):
        if not self.var_configs:
            QMessageBox.information(self, "No variables", "This graph has no variables to remove.")
            return
        var_name, ok = QInputDialog.getItem(self, "Remove variable", "Select variable:", list(self.var_configs.keys()), 0, False)
        if ok and var_name:
            self.remove_variable(var_name)

    def remove_variable(self, var_name):
        if var_name in self.curves:
            self.plot.removeItem(self.curves[var_name])
            del self.curves[var_name]
        if var_name in self.var_configs:
            del self.var_configs[var_name]

    def config_var_dialog(self):
        if not self.var_configs:
            QMessageBox.information(self, "No variables", "This graph has no variables to configure.")
            return
        var_name, ok = QInputDialog.getItem(self, "Configure variable", "Select variable:", list(self.var_configs.keys()), 0, False)
        if ok and var_name:
            cfg = self.var_configs[var_name]
            x, _ = self.variables[var_name]
            dlg = VariableConfigDialog(var_name, x[0], x[-1], existing_cfg=cfg)
            if dlg.exec_() == QDialog.Accepted:
                self.var_configs[var_name] = dlg.get_config()
                self.update_plot()

    def update_plot(self):
        # remove existing curves and replot with configs
        for c in list(self.curves.values()):
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.curves.clear()

        for var_name, cfg in self.var_configs.items():
            x, y = self.variables[var_name]
            arrx = np.array(x)
            arry = np.array(y, dtype=float) * cfg["scale"]
            mask = (arrx >= cfg["time_start"]) & (arrx <= cfg["time_end"])
            xf = arrx[mask]
            yf = arry[mask]
            pen = pg.mkPen(cfg["color"], width=2)
            if cfg["plot_type"] == "line":
                curve = self.plot.plot(xf, yf, pen=pen, name=var_name)
            else:
                curve = self.plot.plot(xf, yf, pen=None, symbol='o', symbolBrush=cfg["color"], name=var_name)
            self.curves[var_name] = curve

    def toggle_pick_mode(self):
        """Toggle crosshair mode (enable/disable)."""
        self.pick_enabled = not self.pick_enabled
        if self.pick_enabled:
            if self.vline not in self.plot.items():
                self.plot.addItem(self.vline)
            if self.text_item not in self.plot.items():
                self.plot.addItem(self.text_item)
            self.pick_toggle_btn = self.sender()
            self.pick_toggle_btn.setText("🔎 Crosshair ON")
            self.pick_toggle_btn.setStyleSheet("background-color: lightgreen;")
            self.text_item.show()
            self.vline.show()
        else:
            try:
                self.plot.removeItem(self.vline)
                self.plot.removeItem(self.text_item)
            except Exception:
                pass
            self.pick_toggle_btn = self.sender()
            self.pick_toggle_btn.setText("🔎 Crosshair OFF")
            self.pick_toggle_btn.setStyleSheet("")
            self.text_item.hide()


    def crosshair_moved(self):
        """Update text label when user drags the crosshair line."""
        if not self.pick_enabled:
            return

        x = self.vline.value()
        self.update_crosshair_text(x)



    def update_crosshair_text(self, x):
        """Display all variable values at the chosen time x."""
        text = f"<b>t = {x:.3f}</b><br>"
        for var, item in self.curves.items():
            if item is not None:
                x_data, y_data = item.getData()
                if len(x_data) > 0:
                    idx = (abs(x_data - x)).argmin()
                    text += f"{var}: {y_data[idx]:.4f}<br>"

        self.text_item.setHtml(text)

        # Keep text visible at the top of the plot
        view_rect = self.plot.viewRect()
        y_top = view_rect.top()
        self.text_item.setPos(x, y_top)



    def move_graph_dialog(self):
        # ask user for target row/col
        rows, cols = self.grid_manager.grid_rows, self.grid_manager.grid_cols
        choices = []
        for r in range(rows):
            for c in range(cols):
                choices.append(f"{r},{c}")
        choice, ok = QInputDialog.getItem(self, "Move Graph", "Choose target cell (row,col):", choices, 0, False)
        if ok and choice:
            r, c = map(int, choice.split(","))
            if self.grid_manager.cell_occupied(r, c):
                QMessageBox.warning(self, "Occupied", f"Cell {r},{c} is occupied.")
                return
            self.grid_manager.move_graph(self, r, c)

    def remove_self(self):
        self.grid_manager.remove_graph(self)


# -------------------------
# Grid manager / Main app
# -------------------------
class GridPlotter(QWidget):
    def __init__(self, rows=3, cols=3):
        super().__init__()
        self.setWindowTitle("CSV Grid Plotter (Crosshair + per-graph config)")
        self.resize(1400, 900)

        self.grid_rows = rows
        self.grid_cols = cols
        # variables: name -> (time_array, value_array)
        self.variables = {}
        self.graphs = {}  # (r,c) -> GraphWidget

        main_layout = QVBoxLayout(self)
        controls = QHBoxLayout()

        self.load_btn = QPushButton("📁 Load CSV")
        self.add_graph_btn = QPushButton("➕ Add Graph")
        self.rows_spin = QSpinBox()
        self.rows_spin.setMinimum(1)
        self.rows_spin.setValue(rows)
        self.cols_spin = QSpinBox()
        self.cols_spin.setMinimum(1)
        self.cols_spin.setValue(cols)
        self.grid_apply_btn = QPushButton("Apply Grid Size")

        controls.addWidget(self.load_btn)
        controls.addWidget(self.add_graph_btn)
        controls.addWidget(QLabel("Rows:"))
        controls.addWidget(self.rows_spin)
        controls.addWidget(QLabel("Cols:"))
        controls.addWidget(self.cols_spin)
        controls.addWidget(self.grid_apply_btn)
        controls.addStretch()

        main_layout.addLayout(controls)

        # variable list on left
        content_layout = QHBoxLayout()
        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QListWidget.MultiSelection)
        content_layout.addWidget(self.var_list, 20)

        # grid layout right
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        content_layout.addWidget(self.grid_widget, 80)

        main_layout.addLayout(content_layout)

        # signals
        self.load_btn.clicked.connect(self.load_csvs)
        self.add_graph_btn.clicked.connect(self.add_graph_dialog)
        self.grid_apply_btn.clicked.connect(self.apply_grid_size)

        self._init_grid_cells()

    def _init_grid_cells(self):
        # clear grid layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # create empty placeholders
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                placeholder = QLabel(f"Cell {r},{c} (empty)")
                placeholder.setStyleSheet("border: 1px dashed gray; min-height: 200px;")
                placeholder.setAlignment(Qt.AlignCenter)
                self.grid_layout.addWidget(placeholder, r, c)

    def apply_grid_size(self):
        new_r = int(self.rows_spin.value())
        new_c = int(self.cols_spin.value())
        # If shrinking, ensure no graphs are in removed cells
        for (r, c) in list(self.graphs.keys()):
            if r >= new_r or c >= new_c:
                QMessageBox.warning(self, "Grid resize", "Cannot shrink grid while graphs occupy cells outside new size.")
                return
        self.grid_rows = new_r
        self.grid_cols = new_c
        self._init_grid_cells()
        # re-add existing graphs
        for (r, c), g in self.graphs.items():
            self.grid_layout.addWidget(g, r, c)

    def load_csvs(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open CSV files", "", "CSV Files (*.csv)")
        for p in paths:
            try:
                df = pd.read_csv(p)
                name = os.path.splitext(os.path.basename(p))[0]
                # detect time
                time_col = None
                for col in df.columns:
                    if col.lower() in ("time", "t"):
                        time_col = col
                        break
                if time_col is None:
                    df["index"] = np.arange(len(df))
                    time_col = "index"
                for col in df.columns:
                    if col == time_col:
                        continue
                    var_name = f"{col}_{name}"
                    x = df[time_col].values.astype(float)
                    y = df[col].values.astype(float)
                    self.variables[var_name] = (x, y)
                    self.var_list.addItem(QListWidgetItem(var_name))
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Failed to load {p}\n{e}")

    def add_graph_dialog(self):
        # ask row,col or auto
        choices = ["Auto place (next free)"]
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                choices.append(f"{r},{c}")
        choice, ok = QInputDialog.getItem(self, "Add Graph", "Choose grid cell or Auto:", choices, 0, False)
        if not ok:
            return
        if choice.startswith("Auto"):
            placed = self.add_graph_auto()
            if not placed:
                QMessageBox.warning(self, "Full", "No free cell available. Remove a graph or enlarge grid.")
        else:
            r, c = map(int, choice.split(","))
            if self.cell_occupied(r, c):
                QMessageBox.warning(self, "Occupied", f"Cell {r},{c} is occupied.")
                return
            self.add_graph(r, c)

    def add_graph_auto(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if not self.cell_occupied(r, c):
                    self.add_graph(r, c)
                    return True
        return False

    def add_graph(self, r, c):
        gw = GraphWidget(self, r, c, (self.grid_rows, self.grid_cols))
        gw.set_variables_source(self.variables)
        self.grid_layout.addWidget(gw, r, c)
        self.graphs[(r, c)] = gw

    def cell_occupied(self, r, c):
        return (r, c) in self.graphs

    def remove_graph(self, graph_widget):
        # find cell
        found = None
        for k, g in self.graphs.items():
            if g is graph_widget:
                found = k
                break
        if not found:
            return
        r, c = found
        self.grid_layout.removeWidget(graph_widget)
        graph_widget.deleteLater()
        del self.graphs[(r, c)]
        # put placeholder back
        placeholder = QLabel(f"Cell {r},{c} (empty)")
        placeholder.setStyleSheet("border: 1px dashed gray; min-height: 200px;")
        placeholder.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(placeholder, r, c)

    def move_graph(self, graph_widget, new_r, new_c):
        # check validity
        if self.cell_occupied(new_r, new_c):
            QMessageBox.warning(self, "Occupied", "Target cell is occupied.")
            return
        # find old
        found = None
        for k, g in self.graphs.items():
            if g is graph_widget:
                found = k
                break
        if not found:
            return
        old_r, old_c = found
        # remove widget from old
        self.grid_layout.removeWidget(graph_widget)
        # replace old with placeholder
        placeholder = QLabel(f"Cell {old_r},{old_c} (empty)")
        placeholder.setStyleSheet("border: 1px dashed gray; min-height: 200px;")
        placeholder.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(placeholder, old_r, old_c)
        del self.graphs[(old_r, old_c)]
        # put graph in new cell
        self.grid_layout.addWidget(graph_widget, new_r, new_c)
        self.graphs[(new_r, new_c)] = graph_widget
        graph_widget.row = new_r
        graph_widget.col = new_c


# -------------------------
# Main
# -------------------------
def main():
    app = QApplication(sys.argv)
    w = GridPlotter(rows=3, cols=3)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
