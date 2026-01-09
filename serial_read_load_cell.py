import sys
import struct
import csv
import os
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import time

CONFIG_FILE = "config.csv"


class SerialReader(QtCore.QThread):
    data_received = QtCore.pyqtSignal(float, float)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.ser = None
        self.running = False
        self.port = None
        self.baudrate = 115200
        self.HEADER = 0x55
        self.TERMINATOR = 0xFF

    def connect_serial(self, port):
        try:
            self.ser = serial.Serial(port, self.baudrate, timeout=1)
            self.port = port
            self.running = True
            self.connection_changed.emit(True)
            self.start()
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Connection Error", str(e))
            self.connection_changed.emit(False)

    def disconnect_serial(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connection_changed.emit(False)

    def run(self):
        while self.running:
            try:
                b = self.ser.read(1)
                if not b or b[0] != self.HEADER:
                    continue
                data = self.ser.read(8)
                if len(data) != 8:
                    continue
                term = self.ser.read(1)
                if len(term) != 1 or term[0] != self.TERMINATOR:
                    continue
                val1, val2 = struct.unpack('<ff', data)
                self.data_received.emit(val1, val2)
            except Exception:
                pass


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Serial Plot + Auto Calib + Config Control")
        self.resize(1300, 900)

        self.serial_thread = SerialReader()
        self.serial_thread.data_received.connect(self.update_data)
        self.serial_thread.connection_changed.connect(self.on_connection_changed)

        layout = QtWidgets.QVBoxLayout(self)

        # === Top control bar ===
        control_layout = QtWidgets.QHBoxLayout()
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()

        self.refresh_btn = QtWidgets.QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setStyleSheet("background-color: red; color: white;")
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.zero1 = QtWidgets.QLineEdit("0")
        self.span1 = QtWidgets.QLineEdit("203891")
        self.zero2 = QtWidgets.QLineEdit("0")
        self.span2 = QtWidgets.QLineEdit("401440")
        for box in [self.zero1, self.span1, self.zero2, self.span2]:
            box.setFixedWidth(100)

        self.save_config_btn = QtWidgets.QPushButton("💾 Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        self.load_config_btn = QtWidgets.QPushButton("📂 Load Config")
        self.load_config_btn.clicked.connect(self.load_config)

        control_layout.addWidget(QtWidgets.QLabel("Port:"))
        control_layout.addWidget(self.port_combo)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.connect_btn)
        control_layout.addStretch()
        control_layout.addWidget(QtWidgets.QLabel("Zero1:"))
        control_layout.addWidget(self.zero1)
        control_layout.addWidget(QtWidgets.QLabel("Span1:"))
        control_layout.addWidget(self.span1)
        control_layout.addWidget(QtWidgets.QLabel("Zero2:"))
        control_layout.addWidget(self.zero2)
        control_layout.addWidget(QtWidgets.QLabel("Span2:"))
        control_layout.addWidget(self.span2)
        control_layout.addWidget(self.save_config_btn)
        control_layout.addWidget(self.load_config_btn)
        layout.addLayout(control_layout)

        # === Auto calibration section ===
        calib_layout = QtWidgets.QHBoxLayout()
        self.sensor_select = QtWidgets.QComboBox()
        self.sensor_select.addItems(["Sensor 1", "Sensor 2"])
        self.calib_zero_target = QtWidgets.QLineEdit("0")
        self.calib_span_target = QtWidgets.QLineEdit("1.0")
        self.calib_zero_btn = QtWidgets.QPushButton("⚙️ Auto Zero Calib")
        self.calib_span_btn = QtWidgets.QPushButton("📏 Auto Span Calib")

        calib_layout.addWidget(QtWidgets.QLabel("Calibrate:"))
        calib_layout.addWidget(self.sensor_select)
        calib_layout.addSpacing(20)
        calib_layout.addWidget(QtWidgets.QLabel("Zero target:"))
        calib_layout.addWidget(self.calib_zero_target)
        calib_layout.addWidget(self.calib_zero_btn)
        calib_layout.addSpacing(20)
        calib_layout.addWidget(QtWidgets.QLabel("Span target:"))
        calib_layout.addWidget(self.calib_span_target)
        calib_layout.addWidget(self.calib_span_btn)
        layout.addLayout(calib_layout)

        self.calib_zero_btn.clicked.connect(self.auto_zero_calib)
        self.calib_span_btn.clicked.connect(self.auto_span_calib)

        # === Plot toolbar ===
        toolbar_layout = QtWidgets.QHBoxLayout()
        zoom_in_btn = QtWidgets.QPushButton("🔍 +")
        zoom_out_btn = QtWidgets.QPushButton("🔍 -")
        auto_btn = QtWidgets.QPushButton("Auto")
        save_btn = QtWidgets.QPushButton("💾 Save Data")
        clear_btn = QtWidgets.QPushButton("🧹 Clear All Data")
        view_data_btn = QtWidgets.QPushButton("📈 View Data (OFF)")

        toolbar_layout.addWidget(zoom_in_btn)
        toolbar_layout.addWidget(zoom_out_btn)
        toolbar_layout.addWidget(auto_btn)
        toolbar_layout.addWidget(save_btn)
        toolbar_layout.addWidget(clear_btn)
        toolbar_layout.addWidget(view_data_btn)
        toolbar_layout.addStretch()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time (s)')

        self.curve1 = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2), name="Value 1")
        self.curve2 = self.plot_widget.plot(pen=pg.mkPen(color='r', width=2), name="Value 2")
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine))
        self.label = pg.TextItem("", anchor=(0, 1))
        self.view_data_mode = False

        layout.addLayout(toolbar_layout)
        layout.addWidget(self.plot_widget)

        # === Data containers ===
        self.max_points = 300
        self.times = deque(maxlen=self.max_points)
        self.values1 = deque(maxlen=self.max_points)
        self.values2 = deque(maxlen=self.max_points)
        self.start_time = time.time()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(30)

        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseClicked, rateLimit=60, slot=self.on_mouse_click)

        # Toolbar actions
        zoom_factor = 1.2
        zoom_in_btn.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((1 / zoom_factor, 1 / zoom_factor)))
        zoom_out_btn.clicked.connect(lambda: self.plot_widget.getViewBox().scaleBy((zoom_factor, zoom_factor)))
        auto_btn.clicked.connect(lambda: self.plot_widget.enableAutoRange())
        save_btn.clicked.connect(self.save_data)
        clear_btn.clicked.connect(self.clear_data)
        view_data_btn.clicked.connect(lambda: self.toggle_view_data_mode(view_data_btn))

        # Auto-load config if exists
        if os.path.exists(CONFIG_FILE):
            self.load_config(CONFIG_FILE)

    # === Core functions ===
    def auto_zero_calib(self):
        sensor = self.sensor_select.currentText()
        avg = self.collect_average()
        if avg is None:
            return
        if sensor == "Sensor 1":
            self.zero1.setText(f"{avg:.6f}")
        else:
            self.zero2.setText(f"{avg:.6f}")
        QtWidgets.QMessageBox.information(self, "Auto Zero", f"{sensor} zero set to {avg:.6f}")

    def auto_span_calib(self):
        sensor = self.sensor_select.currentText()
        target = float(self.calib_span_target.text())
        avg = self.collect_average()
        if avg is None:
            return
        if sensor == "Sensor 1":
            z = float(self.zero1.text())
            span = avg - z
            if span != 0:
                self.span1.setText(f"{span/target:.6f}")
        else:
            z = float(self.zero2.text())
            span = avg - z
            if span != 0:
                self.span2.setText(f"{span/target:.6f}")
        QtWidgets.QMessageBox.information(self, "Auto Span", f"{sensor} span calibrated using {avg:.6f}")

    def collect_average(self):
        data = []
        QtWidgets.QMessageBox.information(self, "Collecting", "Collecting 10 data points...")
        for _ in range(100):
            if len(self.values1) == 0:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.1)
                continue
            data.append((self.values1[-1], self.values2[-1]))
            time.sleep(0.05)
        if not data:
            QtWidgets.QMessageBox.warning(self, "Error", "No data available to calibrate.")
            return None
        avg1 = sum([x[0] for x in data]) / len(data)
        avg2 = sum([x[1] for x in data]) / len(data)
        return avg1 if self.sensor_select.currentText() == "Sensor 1" else avg2

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)

    def toggle_connection(self):
        if self.serial_thread.running:
            self.serial_thread.disconnect_serial()
        else:
            port = self.port_combo.currentText()
            if port:
                self.serial_thread.connect_serial(port)

    def on_connection_changed(self, connected):
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setStyleSheet("background-color: green; color: white;" if connected else "background-color: red; color: white;")

    def update_data(self, val1, val2):
        try:
            z1, s1 = float(self.zero1.text()), float(self.span1.text())
            z2, s2 = float(self.zero2.text()), float(self.span2.text())
            v1, v2 = (val1 - z1) / s1, (val2 - z2) / s2
            t = time.time() - self.start_time
            self.times.append(t)
            self.values1.append(v1)
            self.values2.append(v2)
        except Exception:
            pass

    def refresh_plot(self):
        if len(self.times) > 1:
            self.curve1.setData(list(self.times), list(self.values1))
            self.curve2.setData(list(self.times), list(self.values2))

    def clear_data(self):
        self.times.clear()
        self.values1.clear()
        self.values2.clear()
        self.curve1.clear()
        self.curve2.clear()

    def save_data(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "Value 1", "Value 2"])
                for t, v1, v2 in zip(self.times, self.values1, self.values2):
                    writer.writerow([t, v1, v2])

    def save_config(self):
        with open(CONFIG_FILE, "w", newline="") as f:
            csv.writer(f).writerows([
                ["Port", "Zero1", "Span1", "Zero2", "Span2"],
                [self.port_combo.currentText(), self.zero1.text(), self.span1.text(), self.zero2.text(), self.span2.text()]
            ])
        QtWidgets.QMessageBox.information(self, "Saved", "Configuration saved to config.csv")

    def load_config(self, path=None):
        try:
            path = path or QtWidgets.QFileDialog.getOpenFileName(self, "Load Config", "", "CSV Files (*.csv)")[0]
            if not path:
                return
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                row = next(reader)
            self.zero1.setText(row["Zero1"])
            self.span1.setText(row["Span1"])
            self.zero2.setText(row["Zero2"])
            self.span2.setText(row["Span2"])
            index = self.port_combo.findText(row["Port"])
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
            QtWidgets.QMessageBox.information(self, "Loaded", f"Configuration loaded from {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", str(e))

    def toggle_view_data_mode(self, button):
        self.view_data_mode = not self.view_data_mode
        if self.view_data_mode:
            self.plot_widget.addItem(self.vLine)
            self.plot_widget.addItem(self.hLine)
            self.plot_widget.addItem(self.label)
            button.setText("📈 View Data (ON)")
            button.setStyleSheet("background-color: lightgreen;")
        else:
            self.plot_widget.removeItem(self.vLine)
            self.plot_widget.removeItem(self.hLine)
            self.plot_widget.removeItem(self.label)
            button.setText("📈 View Data (OFF)")
            button.setStyleSheet("")

    def on_mouse_click(self, event):
        if not self.view_data_mode:
            return
        pos = event[0].scenePos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mp = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
            self.vLine.setPos(mp.x())
            self.hLine.setPos(mp.y())
            self.label.setText(f"t={mp.x():.2f}s\nv={mp.y():.6f}")
            self.label.setPos(mp.x(), mp.y())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
