from lightning import LightningApp, LightningFlow, LightningWork
from lightning.components.python.popen import PopenPythonScript


class Main(LightningFlow):
    def __init__(self):
        super().__init__()
        self.script_runner = PopenPythonScript(script_path="training/train.py")

    def run(self):
        self.script_runner.run()


app = LightningApp(Main())
