import os

from lightning import LightningApp, LightningFlow
from lightning.app.frontend.web import StaticWebFrontend


class UI(LightningFlow):
    def configure_layout(self):
        return StaticWebFrontend(os.path.join(os.path.dirname(__file__), f"ui"))


class Main(LightningFlow):
    def __init__(self):
        super().__init__()
        self.ui = UI()
        self.say_hello = True

    def run(self):
        if self.say_hello:
            print("Hello")
            self.say_hello = False

    def configure_layout(self):
        return {"name": "Hello World", "content": self.ui}


app = LightningApp(Main())
